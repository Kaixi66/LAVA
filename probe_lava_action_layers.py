#!/usr/bin/env python3
"""Frozen layer probe for choosing the LAVA action representation.

The LiLa policy and DINOv3 are kept frozen.  A separate, identically
initialized action projector and World Residual encoder are trained for each
candidate action-expert layer.  Validation is split by whole episodes and the
reported controlled retrieval metric only compares examples from the same
task and temporal scale.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataloader.dataset import collate_fn, create_dataset
from models.model_runner import ModelFactory, VLAWrapper
from models.vla_model_fm import (
    WorldResidualEncoder,
    normalized_logsignature,
    sample_full_shuffle_permutation,
)


LOGGER = logging.getLogger("lava_layer_probe")


class ActionProjector(nn.Module):
    def __init__(self, input_dim: int, residual_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, residual_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LayerProbe(nn.Module):
    def __init__(self, action_dim: int, dino_dim: int, residual_dim: int,
                 world_hidden_dim: int, world_layers: int, world_heads: int):
        super().__init__()
        self.action_projector = ActionProjector(action_dim, residual_dim)
        self.world_encoder = WorldResidualEncoder(
            feat_dim=dino_dim,
            residual_dim=residual_dim,
            hidden_dim=world_hidden_dim,
            num_queries=1,
            num_layers=world_layers,
            num_heads=world_heads,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="1,2,3,4,5,6,7,8,9,10,11,12,final")
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--val_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_layer_names(value: str, depth: int) -> list[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Layer list must be non-empty and unique, got {names}")
    for name in names:
        if name == "final":
            continue
        if not name.isdigit() or not 1 <= int(name) <= depth:
            raise ValueError(f"Layer '{name}' must be in [1,{depth}] or 'final'")
    return names


def episode_split_indices(dataset, val_fraction: float, seed: int) -> tuple[list[int], list[int], dict]:
    """Split whole episodes within each task/data-mode stratum."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    groups = defaultdict(list)
    for episode_index, meta in enumerate(dataset.episode_metadata):
        path = str(meta["hdf5_path"])
        mode = "randomized" if "demo_randomized" in path else "clean"
        groups[(meta["task_name"], mode)].append(episode_index)

    rng = random.Random(seed)
    train_episodes, val_episodes = [], []
    for key in sorted(groups):
        episodes = list(groups[key])
        rng.shuffle(episodes)
        count = max(1, int(round(len(episodes) * val_fraction)))
        if len(episodes) > 1:
            count = min(count, len(episodes) - 1)
        val_episodes.extend(episodes[:count])
        train_episodes.extend(episodes[count:])

    def frame_indices(episode_indices: list[int]) -> list[int]:
        result = []
        for episode_index in episode_indices:
            meta = dataset.episode_metadata[episode_index]
            # The final frame cannot provide even a one-transition LAVA path.
            result.extend(range(int(meta["global_start"]), int(meta["global_end"]) - 1))
        return result

    train_indices = frame_indices(train_episodes)
    val_indices = frame_indices(val_episodes)
    if not train_indices or not val_indices:
        raise RuntimeError("Episode split produced an empty train or validation set")
    summary = {
        "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "strata": len(groups),
    }
    return train_indices, val_indices, summary


def balanced_validation_indices(dataset, indices: list[int], sample_count: int,
                                seed: int) -> tuple[list[int], dict]:
    """Select validation frames round-robin across task/data-mode strata.

    The dataset is stored task-by-task, so taking the first N validation frames
    silently evaluates only the first task.  This deterministic selector gives
    every available (task, clean/randomized) stratum the same quota (within one
    frame) before shuffling the final order consumed by the DataLoader.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    groups = defaultdict(list)
    episode_ends = np.asarray(dataset._ep_end_bounds)
    for frame_index in indices:
        episode_index = int(np.searchsorted(episode_ends, frame_index, side="right"))
        meta = dataset.episode_metadata[episode_index]
        path = str(meta["hdf5_path"])
        mode = "randomized" if "demo_randomized" in path else "clean"
        groups[(str(meta["task_name"]), mode)].append(frame_index)
    if not groups:
        raise RuntimeError("No validation strata were found")

    rng = random.Random(seed)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    target = min(sample_count, sum(len(values) for values in groups.values()))
    selected = []
    offsets = {key: 0 for key in keys}
    while len(selected) < target:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                offsets[key] = offset + 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    selected_counts = {
        f"{task}/{mode}": offsets[(task, mode)] for task, mode in keys
    }
    task_counts = defaultdict(int)
    for (task, mode), count in offsets.items():
        task_counts[task] += count
    summary = {
        "requested_frames": sample_count,
        "selected_frames": len(selected),
        "strata": len(keys),
        "tasks": len(task_counts),
        "min_frames_per_stratum": min(selected_counts.values()),
        "max_frames_per_stratum": max(selected_counts.values()),
        "min_frames_per_task": min(task_counts.values()),
        "max_frames_per_task": max(task_counts.values()),
        "frames_per_stratum": selected_counts,
        "frames_per_task": dict(sorted(task_counts.items())),
    }
    return selected, summary


def make_loader(dataset, indices: list[int], batch_size: int, workers: int,
                shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        # NFS-backed multiprocessing temp directories can keep a completed
        # Slurm allocation alive during persistent-worker finalization.
        persistent_workers=False,
        collate_fn=collate_fn,
        drop_last=shuffle,
        generator=generator,
    )


def clone_probe_initialization(probes: nn.ModuleDict) -> None:
    """Give every candidate exactly the same probe initialization."""
    first = next(iter(probes.values())).state_dict()
    for probe in list(probes.values())[1:]:
        probe.load_state_dict(copy.deepcopy(first), strict=True)


def add_time_and_signature(path: torch.Tensor, depth: int = 2) -> torch.Tensor:
    length = path.shape[0]
    time = torch.full((length, 1), 1.0 / length, device=path.device, dtype=path.dtype)
    return F.normalize(normalized_logsignature(torch.cat((path, time), dim=-1), depth), dim=0)


def make_order_permutations(scales: torch.Tensor) -> list[torch.Tensor | None]:
    """Generate one common set of hard negatives for every layer candidate."""
    return [
        None if int(scale) < 2 else sample_full_shuffle_permutation(
            int(scale), device=scales.device)
        for scale in scales.tolist()
    ]


def encode_probe_paths(probe: LayerProbe, action_hidden: torch.Tensor,
                       world_differences: list[torch.Tensor], batch_indices: torch.Tensor,
                       starts: torch.Tensor, scales: torch.Tensor,
                       order_negative: bool = True,
                       order_permutations: list[torch.Tensor | None] | None = None):
    lengths = [int(value) for value in scales.tolist()]
    device = action_hidden.device
    flat_world = torch.cat([value.to(device) for value in world_differences], dim=0)
    world_flat = probe.world_encoder(flat_world)
    world_paths = list(world_flat.split(lengths, dim=0))

    action_signatures, world_signatures = [], []
    shuffled_signatures, shuffled_rows = [], []
    if order_permutations is not None and len(order_permutations) != len(lengths):
        raise ValueError("order_permutations must match the number of sampled paths")
    for row, (batch_index, start, scale, world_path) in enumerate(zip(
            batch_indices.tolist(), starts.tolist(), lengths, world_paths)):
        hidden_path = action_hidden[batch_index, start + 1:start + scale + 1]
        if hidden_path.shape[0] != scale:
            raise RuntimeError(f"Invalid action alignment at [{start + 1}:{start + scale + 1}]")
        action_path = probe.action_projector(hidden_path)
        action_signatures.append(add_time_and_signature(action_path))
        world_signatures.append(add_time_and_signature(world_path))
        if order_negative and scale >= 2:
            permutation = (order_permutations[row] if order_permutations is not None
                           else sample_full_shuffle_permutation(scale, device=device))
            if permutation is None:
                raise RuntimeError("Missing order permutation for a multi-step path")
            shuffled_signatures.append(add_time_and_signature(world_path[permutation]))
            shuffled_rows.append(row)

    return (
        F.normalize(torch.stack(action_signatures), dim=-1),
        F.normalize(torch.stack(world_signatures), dim=-1),
        F.normalize(torch.stack(shuffled_signatures), dim=-1) if shuffled_signatures else None,
        torch.tensor(shuffled_rows, device=device, dtype=torch.long),
    )


def probe_loss(action_sig: torch.Tensor, world_sig: torch.Tensor,
               shuffled_sig: torch.Tensor | None, shuffled_rows: torch.Tensor,
               temperature: float) -> tuple[torch.Tensor, dict]:
    ordinary = action_sig @ world_sig.T
    logits = ordinary
    labels = torch.arange(action_sig.shape[0], device=action_sig.device)
    if shuffled_sig is not None:
        shuffle_logits = action_sig @ shuffled_sig.T
        logits = torch.cat((ordinary, shuffle_logits), dim=1)
        paired_shuffle = shuffle_logits[shuffled_rows, torch.arange(
            shuffled_rows.numel(), device=action_sig.device)]
        paired_positive = ordinary.diagonal()[shuffled_rows]
        order_margin = (paired_positive - paired_shuffle).mean()
    else:
        order_margin = ordinary.new_tensor(float("nan"))
    loss = F.cross_entropy(logits / temperature, labels)
    return loss, {
        "loss": float(loss.detach()),
        "pos_sim": float(ordinary.diagonal().mean().detach()),
        "retrieval_acc": float((logits.argmax(dim=1) == labels).float().mean().detach()),
        "order_margin": float(order_margin.detach()),
    }


@torch.no_grad()
def frozen_backbone_forward(wrapper: VLAWrapper, batch: dict, captures: dict,
                            layer_names: list[str]):
    features = wrapper.get_vision_features(batch["pixel_values"])
    x1 = wrapper.normalize_action(batch["action_sequence"].to(wrapper.device, wrapper.dtype))
    state = batch["state"].to(wrapper.device, wrapper.dtype)
    if state.ndim == 2:
        state = state.unsqueeze(1)
    state = wrapper.normalize_state(state)
    task_cond = batch.get("task_cond")
    if task_cond is not None:
        task_cond = task_cond.to(wrapper.device, wrapper.dtype)
    batch_size = x1.shape[0]
    t = torch.sigmoid(torch.randn(
        batch_size, device=wrapper.device, dtype=wrapper.dtype))
    x0 = torch.randn_like(x1)
    noisy = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    captures.clear()
    output = wrapper.action_model(
        t, noisy_actions=noisy, qpos_history=state,
        dino_features_list=features, task_cond=task_cond)
    hidden = {}
    for name in layer_names:
        if name == "final":
            hidden[name] = output["action_hidden"].detach()
        else:
            hidden[name] = captures[name][:, :wrapper.action_model.action_len].detach()
    world = wrapper.get_evolution_feature_differences(batch["evolution_pixel_values"])
    return hidden, world


def register_layer_hooks(action_model, layer_names: list[str], captures: dict):
    handles = []
    for name in layer_names:
        if name == "final":
            continue
        block = action_model.blocks[int(name) - 1]
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output, key=name: captures.__setitem__(key, output.detach())))
    return handles


def safe_mean(values) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def masked_mean(matrix: torch.Tensor, mask: torch.Tensor) -> float:
    return float(matrix[mask].mean()) if mask.any() else float("nan")


def validation_metrics(action: torch.Tensor, world: torch.Tensor,
                       shuffled: torch.Tensor, shuffled_valid: torch.Tensor,
                       tasks: list[str], scales: torch.Tensor, temperature: float) -> dict:
    action, world = F.normalize(action.float(), dim=-1), F.normalize(world.float(), dim=-1)
    similarity = action @ world.T
    count = similarity.shape[0]
    labels = torch.arange(count, device=similarity.device)
    eye = torch.eye(count, device=similarity.device, dtype=torch.bool)
    task_to_id = {task: index for index, task in enumerate(sorted(set(tasks)))}
    task_ids = torch.tensor([task_to_id[task] for task in tasks], device=similarity.device)
    same_task = task_ids[:, None] == task_ids[None, :]
    same_scale = scales[:, None] == scales[None, :]

    ordinary_acc = (similarity.argmax(dim=1) == labels).float().mean()
    controlled_mask = same_task & same_scale
    controlled_logits = similarity.masked_fill(~controlled_mask, -torch.inf)
    controlled_acc = (controlled_logits.argmax(dim=1) == labels).float().mean()
    same_task_negative = same_task & ~eye
    cross_task_negative = ~same_task
    same_scale_negative = same_scale & ~eye
    cross_scale_negative = ~same_scale

    paired_shuffle = torch.full((count,), torch.nan, device=similarity.device)
    valid_rows = shuffled_valid.nonzero(as_tuple=False).flatten()
    if valid_rows.numel():
        paired_shuffle[valid_rows] = (action[valid_rows] * F.normalize(
            shuffled[valid_rows].float(), dim=-1)).sum(dim=-1)
    positive = similarity.diagonal()
    order_mask = torch.isfinite(paired_shuffle)
    order_margin = positive[order_mask] - paired_shuffle[order_mask]

    return {
        "val_loss": float(F.cross_entropy(similarity / temperature, labels)),
        "retrieval_acc": float(ordinary_acc),
        "controlled_same_task_scale_retrieval_acc": float(controlled_acc),
        "pos_sim": float(positive.mean()),
        "negative_sim": masked_mean(similarity, ~eye),
        "same_task_negative_sim": masked_mean(similarity, same_task_negative),
        "cross_task_negative_sim": masked_mean(similarity, cross_task_negative),
        "task_shortcut_gap": masked_mean(similarity, same_task_negative) - masked_mean(
            similarity, cross_task_negative),
        "same_scale_negative_sim": masked_mean(similarity, same_scale_negative),
        "cross_scale_negative_sim": masked_mean(similarity, cross_scale_negative),
        "scale_shortcut_gap": masked_mean(similarity, same_scale_negative) - masked_mean(
            similarity, cross_scale_negative),
        "shuffle_sim": safe_mean(paired_shuffle[order_mask].tolist()),
        "order_margin": safe_mean(order_margin.tolist()),
        "order_accuracy": safe_mean((order_margin > 0).float().tolist()),
        "samples": count,
        "unique_tasks": len(task_to_id),
        "min_samples_per_task": min(tasks.count(task) for task in task_to_id),
        "max_samples_per_task": max(tasks.count(task) for task in task_to_id),
    }


@torch.no_grad()
def validate(wrapper, probes, loader, captures, layer_names, val_samples, temperature):
    for probe in probes.values():
        probe.eval()
    collected = {name: {"action": [], "world": [], "shuffle": [], "valid": []}
                 for name in layer_names}
    tasks, scales_all = [], []
    seen = 0
    for batch in loader:
        if batch is None or batch.get("evolution_pixel_values") is None:
            continue
        hidden, differences = frozen_backbone_forward(wrapper, batch, captures, layer_names)
        indices = batch["evolution_batch_indices"].to(wrapper.device)
        starts = batch["evolution_starts"].to(wrapper.device)
        scales = batch["evolution_scales"].to(wrapper.device)
        selected_tasks = [batch["task_name"][index] for index in indices.tolist()]
        remaining = val_samples - seen
        if remaining <= 0:
            break
        if indices.numel() > remaining:
            indices, starts, scales = indices[:remaining], starts[:remaining], scales[:remaining]
            differences = differences[:remaining]
            selected_tasks = selected_tasks[:remaining]
        order_permutations = make_order_permutations(scales)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for name, probe in probes.items():
                a, w, s, rows = encode_probe_paths(
                    probe, hidden[name], differences, indices, starts, scales,
                    order_permutations=order_permutations)
                full_shuffle = torch.zeros_like(a)
                valid = torch.zeros(a.shape[0], device=a.device, dtype=torch.bool)
                if s is not None:
                    full_shuffle[rows] = s
                    valid[rows] = True
                collected[name]["action"].append(a.cpu())
                collected[name]["world"].append(w.cpu())
                collected[name]["shuffle"].append(full_shuffle.cpu())
                collected[name]["valid"].append(valid.cpu())
        tasks.extend(selected_tasks)
        scales_all.append(scales.cpu())
        seen += indices.numel()
        if seen >= val_samples:
            break

    if seen == 0:
        raise RuntimeError("Validation produced no LAVA samples")
    scales = torch.cat(scales_all).cuda()
    result = {}
    for name in layer_names:
        values = collected[name]
        result[name] = validation_metrics(
            torch.cat(values["action"]).cuda(), torch.cat(values["world"]).cuda(),
            torch.cat(values["shuffle"]).cuda(), torch.cat(values["valid"]).cuda(),
            tasks, scales, temperature)
    return result


def save_results(output_dir: Path, args, split_summary: dict, results: dict,
                 probes: nn.ModuleDict, checkpoint_epoch) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda name: (
        -results[name]["controlled_same_task_scale_retrieval_acc"],
        -results[name]["order_accuracy"], results[name]["val_loss"]))
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "candidate_layers": list(results),
        "ranking_metric": "controlled_same_task_scale_retrieval_acc",
        "ranking": ranked,
        "best_layer": ranked[0],
        "split": split_summary,
        "settings": vars(args),
        "results": results,
    }
    with (output_dir / "layer_probe_results.json").open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    fields = ["rank", "layer", *next(iter(results.values())).keys()]
    with (output_dir / "layer_probe_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, name in enumerate(ranked, 1):
            writer.writerow({"rank": rank, "layer": name, **results[name]})
    torch.save({"probe_state_dict": probes.state_dict(), "results": results},
               output_dir / "layer_probe_weights.pt")
    with (output_dir / "BEST_LAYER.txt").open("w") as handle:
        handle.write(ranked[0] + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires one CUDA GPU")
    if args.train_steps < 1 or args.val_samples < 1:
        raise ValueError("train_steps and val_samples must be positive")
    set_seed(args.seed)
    device, dtype = torch.device("cuda"), torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config)
    if args.set:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    # Keep checkpoint architecture unchanged, but avoid unused future-frame I/O.
    dataset_config = copy.deepcopy(config)
    dataset_config.model.future_feat.enabled = False
    if "lava" not in dataset_config.model:
        dataset_config.model.lava = {}
    dataset_config.model.lava.enabled = True
    dataset_config.training.lava_sample_ratio = 1.0
    dataset_config.dataset.image_aug = False

    dataset = create_dataset(dataset_config, val=False)
    train_indices, val_indices, split_summary = episode_split_indices(
        dataset, args.val_fraction, args.seed)
    val_indices, validation_selection = balanced_validation_indices(
        dataset, val_indices, args.val_samples, args.seed + 1)
    split_summary["validation_selection"] = validation_selection
    LOGGER.info("Episode split: %s", split_summary)
    train_loader = make_loader(dataset, train_indices, args.batch_size,
                               args.num_workers, True, args.seed)
    val_loader = make_loader(dataset, val_indices, args.batch_size,
                             args.num_workers, False, args.seed + 1)

    vision, dino_dim, registers, patch_size = ModelFactory.create_vision_encoder(
        config.model.vision_encoder.checkpoint_path, dtype=dtype, device=device)
    action_model = ModelFactory.create_action_model(
        config, dino_hidden_size=dino_dim,
        num_dino_layers=len(config.model.vision_encoder.feat_layers),
        task_cond_dim=dino_dim if config.model.get("use_task_cond", False) else None,
        patch_size=patch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    action_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    action_model.to(device=device, dtype=dtype).eval()
    for parameter in action_model.parameters():
        parameter.requires_grad_(False)

    layer_names = parse_layer_names(args.layers, len(action_model.blocks))
    lava_cfg = config.model.get("lava", {})
    probes = nn.ModuleDict({name: LayerProbe(
        action_dim=action_model.hidden_dim,
        dino_dim=dino_dim,
        residual_dim=int(lava_cfg.get("residual_dim", 32)),
        world_hidden_dim=int(lava_cfg.get("qformer", {}).get("hidden_dim", 256)),
        world_layers=int(lava_cfg.get("qformer", {}).get("num_layers", 2)),
        world_heads=int(lava_cfg.get("qformer", {}).get("num_heads", 4)),
    ) for name in layer_names})
    clone_probe_initialization(probes)
    probes.to(device)

    wrapper = VLAWrapper(
        vision_encoder=vision, action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=list(config.model.vision_encoder.feat_layers),
        include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=registers, device=device, dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        norm_stats_key=config.dataset.get("norm_stats_key", "robotwin2"),
        train_config=None,
        lava_target_layer=int(lava_cfg.get("dino_target_layer", -4)),
        vision_encode_batch_size=config.model.vision_encoder.get("encode_batch_size", None),
    ).to(device)
    wrapper.eval()

    captures = {}
    handles = register_layer_hooks(action_model, layer_names, captures)
    optimizer = AdamW(probes.parameters(), lr=args.learning_rate,
                      betas=(0.9, 0.99), weight_decay=args.weight_decay)
    history_path = output_dir / "train_probe.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "layer", "loss", "pos_sim", "retrieval_acc", "order_margin"])

    LOGGER.info("Frozen candidates=%s; trainable probe parameters=%d", layer_names,
                sum(parameter.numel() for parameter in probes.parameters()))
    iterator = iter(train_loader)
    try:
        for step in range(1, args.train_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            if batch is None or batch.get("evolution_pixel_values") is None:
                continue
            hidden, differences = frozen_backbone_forward(
                wrapper, batch, captures, layer_names)
            indices = batch["evolution_batch_indices"].to(device)
            starts = batch["evolution_starts"].to(device)
            scales = batch["evolution_scales"].to(device)
            order_permutations = make_order_permutations(scales)
            optimizer.zero_grad(set_to_none=True)
            metrics = {}
            total_loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for name, probe in probes.items():
                    encoded = encode_probe_paths(
                        probe, hidden[name], differences, indices, starts, scales,
                        order_permutations=order_permutations)
                    loss, metric = probe_loss(*encoded, args.temperature)
                    total_loss = total_loss + loss
                    metrics[name] = metric
                total_loss = total_loss / len(probes)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
            optimizer.step()

            with history_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                for name in layer_names:
                    metric = metrics[name]
                    writer.writerow([step, name, metric["loss"], metric["pos_sim"],
                                     metric["retrieval_acc"], metric["order_margin"]])
            if step == 1 or step % args.log_interval == 0:
                text = " | ".join(
                    f"{name}:loss={metrics[name]['loss']:.3f},ret={metrics[name]['retrieval_acc']:.3f}"
                    for name in layer_names)
                LOGGER.info("step %d/%d | %s", step, args.train_steps, text)

        results = validate(wrapper, probes, val_loader, captures, layer_names,
                           args.val_samples, args.temperature)
        save_results(output_dir, args, split_summary, results, probes,
                     checkpoint.get("epoch"))
        ranked = sorted(results, key=lambda name: (
            -results[name]["controlled_same_task_scale_retrieval_acc"],
            -results[name]["order_accuracy"], results[name]["val_loss"]))
        LOGGER.info("Final ranking (controlled retrieval):")
        for rank, name in enumerate(ranked, 1):
            metric = results[name]
            LOGGER.info(
                "%d. %-5s controlled=%.4f ordinary=%.4f order_acc=%.4f "
                "order_margin=%.4f task_gap=%.4f scale_gap=%.4f",
                rank, name, metric["controlled_same_task_scale_retrieval_acc"],
                metric["retrieval_acc"], metric["order_accuracy"],
                metric["order_margin"], metric["task_shortcut_gap"],
                metric["scale_shortcut_gap"])
        LOGGER.info("Results saved to %s", output_dir)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()

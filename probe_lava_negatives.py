#!/usr/bin/env python3
"""Fixed-representation negative interventions; no optimizer or weight updates.

Uses real mixed-task batches and shared frozen DINO features/noise/permutations
for all checkpoints. Reports probability mass and gradients at normalized
signature outputs; these are not full-network parameter gradients or SR effects.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataloader.dataset import LAVABatchScaleBatchSampler, collate_fn, create_dataset
from models.model_runner import ModelFactory, VLAWrapper
from models.vla_model_fm import calc_flow_matching_loss


VARIANTS = {"all_sum": (True, False), "all_mean": (True, True),
            "cross_sum": (False, False), "cross_mean": (False, True)}


def capture_signatures(model, forward):
    captured = {}
    code = model.compute_lava_loss.__func__.__code__

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is code:
            local = frame.f_locals
            for name in ("action_signatures", "world_signatures",
                         "temporal_negative_signatures", "far_negative_signatures",
                         "block_swap_signatures", "derangement_signatures",
                         "block_action_indices", "derangement_action_indices",
                         "interval_scales", "selected_task_names"):
                value = local[name]
                if torch.is_tensor(value):
                    value = value.detach()
                elif isinstance(value, list):
                    value = [x.detach() if torch.is_tensor(x) else x for x in value]
                captured[name] = value
            captured["reference_loss"] = float(arg[0])

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        with torch.no_grad():
            forward()
    finally:
        sys.setprofile(previous)
    if not captured:
        raise RuntimeError("No LAVA signature capture; supervision did not execute")
    return captured


def tangent(gradient, value):
    return gradient - (gradient * value).sum(-1, keepdim=True) * value / value.square().sum(-1, keepdim=True).clamp_min(1e-12)


def compare_negatives(captured, temperature=.07, amp=False, world_tasks=()):
    action = captured["action_signatures"].float().detach().requires_grad_()
    world = captured["world_signatures"].float().detach().requires_grad_()
    def tensor(value):
        return value if torch.is_tensor(value) else torch.stack(value)
    local = tensor(captured["temporal_negative_signatures"]).float().detach().requires_grad_()
    far = tensor(captured["far_negative_signatures"]).float().detach().requires_grad_()
    tasks = captured["selected_task_names"]
    scales = captured["interval_scales"]
    n, dim = action.shape
    device = action.device
    order_inputs = []
    order_indices = []
    for signatures, indices in (("block_swap_signatures", "block_action_indices"),
                                 ("derangement_signatures", "derangement_action_indices")):
        values = captured[signatures]
        order_inputs.append((torch.stack(values) if values else action.new_empty(0, dim)).float().detach().requires_grad_())
        order_indices.append(torch.tensor(captured[indices], device=device, dtype=torch.long))
    world_inputs = (world, local, far, *order_inputs)
    same_task = torch.tensor([[a == b for b in tasks] for a in tasks], device=device)
    mask_all = ~torch.eye(n, dtype=torch.bool, device=device)
    mask_all &= scales[:, None] == scales[None, :]
    baseline_action_grad = None
    baseline_world_grad = {}
    output = []
    for variant, (include_same, average) in VARIANTS.items():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            batch_scores = (action @ world.T).float() / temperature
            positive = (action * world).sum(-1) / temperature
            local_scores = (action * local).sum(-1) / temperature
            far_scores = (action * far).sum(-1) / temperature
            order_scores = []
            for signatures, indices in zip(order_inputs, order_indices):
                scores = action.new_full((n,), -torch.inf)
                if indices.numel():
                    scores = scores.index_copy(0, indices, (action[indices] * signatures).sum(-1) / temperature)
                order_scores.append(scores)
            order_scores = torch.stack(order_scores, dim=1)
            order_counts = torch.isfinite(order_scores).sum(1)
            safe_order = order_scores.masked_fill((order_counts == 0)[:, None], 0.)
            order = (torch.logsumexp(safe_order, dim=1) - order_counts.clamp_min(1).float().log()).masked_fill(order_counts == 0, -torch.inf)
            mask = mask_all if include_same else mask_all & ~same_task
            counts = mask.sum(1)
            batch_scores = batch_scores.masked_fill(~mask, -torch.inf)
            if average:
                batch_scores = batch_scores - counts.clamp_min(1).float().log()[:, None]
            logits = torch.cat((positive[:, None], batch_scores, local_scores[:, None],
                                far_scores[:, None], order[:, None]), dim=1)
            losses = F.cross_entropy(logits, torch.zeros(n, device=device, dtype=torch.long), reduction="none")
        if variant == "all_sum":
            error = abs(float(losses.mean().detach()) - captured["reference_loss"])
            if error > 3e-4:
                raise RuntimeError(f"Replay does not match production loss: absolute error={error}")
        gradient = torch.autograd.grad(losses.sum(), action, retain_graph=True)[0]
        gradient = tangent(gradient, action.detach())
        if baseline_action_grad is None:
            baseline_action_grad = gradient.detach()
        probs = logits.detach().softmax(1)
        batch_probs = probs[:, 1:1+n]
        ratios = gradient.norm(dim=1) / baseline_action_grad.norm(dim=1).clamp_min(1e-12)
        cosines = F.cosine_similarity(gradient, baseline_action_grad, dim=1)
        for i in range(n):
            row = {
                "variant": variant, "anchor": i, "task": tasks[i], "scale": int(scales[i]),
                "loss": float(losses[i].detach()), "batch_candidates": int(counts[i]),
                "p_positive": float(probs[i, 0]), "p_batch": float(batch_probs[i].sum()),
                "p_same_task": float(batch_probs[i][same_task[i]].sum()),
                "p_local": float(probs[i, -3]), "p_far": float(probs[i, -2]),
                "p_order": float(probs[i, -1]),
                "action_grad_norm": float(gradient[i].norm()),
                "action_grad_ratio_to_all_sum": float(ratios[i]),
                "action_grad_cos_to_all_sum": float(cosines[i]),
            }
            if tasks[i] in world_tasks:
                grads = torch.autograd.grad(losses[i], world_inputs, allow_unused=True, retain_graph=True)
                flat = torch.cat([(tangent(g, value.detach()) if g is not None else torch.zeros_like(value)).flatten()
                                  for g, value in zip(grads, world_inputs)])
                if variant == "all_sum":
                    baseline_world_grad[i] = flat.detach()
                original = baseline_world_grad[i]
                row.update(world_signature_grad_norm=float(flat.norm()),
                           world_signature_grad_ratio_to_all_sum=float(flat.norm() / original.norm().clamp_min(1e-12)),
                           world_signature_grad_cos_to_all_sum=float(F.cosine_similarity(flat[None], original[None])))
            if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise RuntimeError(f"Non-finite negative diagnostic: {row}")
            output.append(row)
    return output


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint"], row["variant"], row["task"], row["scale"])].append(row)
    results = []
    for group, values in sorted(grouped.items()):
        summary = dict(zip(("checkpoint", "variant", "task", "scale"), group))
        summary["n"] = len(values)
        for key in values[0]:
            if key.startswith(("p_", "action_grad_", "world_signature_grad_")) or key == "loss":
                nums = [x[key] for x in values if key in x]
                summary[key] = {"mean": float(np.mean(nums)), "median": float(np.median(nums)),
                                "p10": float(np.quantile(nums, .1)), "p90": float(np.quantile(nums, .9))}
        results.append(summary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=/absolute/checkpoint.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6202)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    specs = [item.split("=", 1) for item in args.checkpoint]
    config = OmegaConf.load(Path(specs[0][1]).parent / "config.yaml")
    vision, dino_dim, registers, patch_size = ModelFactory.create_vision_encoder(
        config.model.vision_encoder.checkpoint_path, dtype=torch.bfloat16, device="cuda")
    models = []
    for label, path in specs:
        cfg = OmegaConf.load(Path(path).parent / "config.yaml")
        model = ModelFactory.create_action_model(cfg, dino_dim, len(cfg.model.vision_encoder.feat_layers),
                                                task_cond_dim=dino_dim, patch_size=patch_size)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        del saved
        # All interventions fix action weighting OFF. Original checkpoints are
        # loaded strictly first; their learned encoders are left unchanged.
        model.lava_action_similarity_weighting = False
        model.requires_grad_(False).to(device="cuda", dtype=torch.bfloat16).eval()
        models.append((label, model))
        gc.collect()
    wrapper = VLAWrapper(
        vision, models[0][1], config.training.time_sampler,
        list(config.model.vision_encoder.feat_layers), config.model.vision_encoder.include_cls_register,
        registers, "cuda", torch.bfloat16,
        norm_stats_path="/fs/cml-projects/WAM/data/robotwin_200_10_assets/stat-local-200-10.json",
        lava_target_layer=int(config.model.lava.dino_target_layer),
        vision_encode_batch_size=32).cuda().eval()
    dataset_config = copy.deepcopy(config)
    dataset_config.model.future_feat.enabled = False
    dataset_config.training.lava_action_similarity_weighting = False
    dataset_config.training.lava_action_component_calibration = False
    dataset_config.training.lava_negative_mode = "mixed_batch"
    dataset = create_dataset(dataset_config, val=False)
    sampler = LAVABatchScaleBatchSampler(len(dataset), 128, config.training.lava_scales, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                        collate_fn=collate_fn, pin_memory=True)
    metadata = {"checkpoints": specs, "seed": args.seed, "batches": args.batches,
                "variants": VARIANTS, "action_weighting": False,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "interpretation": "Fixed-representation probability mass and tangent gradients at unit signatures. No parameter updates. These are mechanism diagnostics, not causal SR estimates."}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows = []
    with (out / "observations.jsonl").open("w") as output:
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            n = len(batch.get("evolution_pixel_values") or [])
            if not n:
                raise RuntimeError("Diagnostic batch has no LAVA anchors")
            with torch.no_grad():
                vision_features = wrapper.get_vision_features(batch["pixel_values"])
                paths = wrapper.get_evolution_feature_differences(
                    batch["evolution_pixel_values"] + batch["temporal_negative_pixel_values"] + batch["far_negative_pixel_values"])
                actions = wrapper.normalize_action(batch["action_sequence"].cuda().to(torch.bfloat16))
                state = wrapper.normalize_state(batch["state"].cuda().to(torch.bfloat16))
                if state.ndim == 2:
                    state = state[:, None]
                task_cond = batch["task_cond"].cuda().to(torch.bfloat16)
            rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            for label, model in models:
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_cuda)
                def forward():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return calc_flow_matching_loss(
                            model, actions, dino_features_list=vision_features, qpos_history=state,
                            task_cond=task_cond, time_sampler=config.training.time_sampler,
                            time_mu=config.training.time_mu, time_sigma=config.training.time_sigma,
                            use_lava=True, lambda_lava=.01, world_feature_differences=paths[:n],
                            temporal_negative_feature_differences=paths[n:2*n], far_negative_feature_differences=paths[2*n:],
                            lava_batch_indices=batch["evolution_batch_indices"], lava_interval_starts=batch["evolution_starts"],
                            lava_interval_scales=batch["evolution_scales"], lava_negative_mode="mixed_batch",
                            task_names=batch["task_name"])
                capture = capture_signatures(model, forward)
                rows = compare_negatives(capture, amp=True, world_tasks=("hanging_mug", "open_microwave"))
                for row in rows:
                    row.update(checkpoint=label, batch=batch_index)
                    output.write(json.dumps(row) + "\n")
                all_rows.extend(rows)
                output.flush()
                print(f"batch={batch_index + 1}/{args.batches} checkpoint={label} anchors={n} scale={batch['evolution_scales'][0].item()}", flush=True)
            (out / "status.json").write_text(json.dumps({"completed_batches": batch_index + 1, "rows": len(all_rows)}))
    (out / "summary.json").write_text(json.dumps(summarize(all_rows), indent=2) + "\n")
    (out / "complete.json").write_text(json.dumps({"completed_batches": args.batches, "rows": len(all_rows)}))


if __name__ == "__main__":
    main()

# Robotwin2 Dataset Loader for VLA-DINOv3 (HDF5 Version)
# Supports Robotwin2 data with HDF5 storage and flexible indexing
import os
import random
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

NUM_THREADS = os.cpu_count() or 4

# Task subset used for the RoboTwin ablations in LiLa-WAM, Section 4.2.
ROBOTWIN_TASK_SETS = {
    "10": (
        "adjust_bottle",
        "grab_roller",
        "hanging_mug",
        "move_stapler_pad",
        "open_microwave",
        "press_stapler",
        "scan_object",
        "stack_blocks_two",
        "stamp_seal",
        "turn_switch",
    ),
}

# ImageNet normalization (DINOv3 uses ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode(buf):
    """Decode bytes to RGB numpy array (cv2 returns BGR; data already saved as RGB-style)."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # NOTE: robotwin data does not need a BGR2RGB conversion
    return img


def _normalize_image(img_np: np.ndarray) -> np.ndarray:
    """
    HxWx3 uint8 (RGB) → 3xHxW float32, ImageNet normalized.
    """
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


class RobotWinTaskDataset(data.Dataset):
    def __init__(self, dataset_dir, data_mode="clean",
                 indices_config=None, camera_names=None, image_size=(320, 240),
                 val=False, image_aug=False,
                 task_cond_dir=None,
                 use_future_feat=False, future_frame_offset=None,
                 task_set="50",
                 use_lava=False, lava_scales=None, lava_sample_ratio=0.25,
                 lava_scale_sampling="uniform", lava_scale_probs=None,
                 lava_sampling_balance="none"):
        """
        RobotWin Dataset for DINOv3-based VLA (no language input).

        Args:
            dataset_dir: dataset root directory
            data_mode: "clean" / "randomized" / "both"
            indices_config: index config {state_indices, action_indices, camera_indices}
            camera_names: camera list (the [0] primary camera is used)
            image_size: (W, H); DINOv3 requires H/W to be multiples of patch_size (=16)
            val: whether this is the validation set
            image_aug: whether to apply color augmentation
            task_cond_dir: directory of precomputed task condition vectors (mirrors the
                           dataset task structure); None = task condition disabled
            use_future_feat: whether to also return a future frame (future-feature
                             prediction training)
            future_frame_offset: offset of the future frame relative to the current
                                 anchor; None -> = chunk_size
            task_set: "50" / "all" for all tasks, or "10" for the paper's
                      fixed 10-task list (the dataset root controls episodes)
            use_lava: return a sampled multi-scale frame evolution path
            lava_scales: candidate temporal intervals in action-token steps
            lava_sample_ratio: probability that a dataset sample receives LAVA supervision
            lava_scale_sampling: "uniform" or "weighted"
            lava_scale_probs: probabilities corresponding to lava_scales in weighted mode
            lava_sampling_balance: "none" preserves frame-uniform sampling; "task"
                                   equalizes expected LAVA samples across tasks; and
                                   "task_episode" also equalizes episodes within each task
        """
        if indices_config is None:
            raise ValueError("indices_config is required")
        if 'state_indices' not in indices_config or 'action_indices' not in indices_config:
            raise ValueError("indices_config missing required keys")

        self.dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, str) else [Path(p) for p in dataset_dir]
        self.data_mode = data_mode
        self.all_episodes = []

        self.task_set = str(task_set).lower()
        if self.task_set in {"50", "all"}:
            self.selected_tasks = None
        elif self.task_set in ROBOTWIN_TASK_SETS:
            self.selected_tasks = frozenset(ROBOTWIN_TASK_SETS[self.task_set])
        else:
            choices = ", ".join(["50", "all", *sorted(ROBOTWIN_TASK_SETS)])
            raise ValueError(f"Unknown RoboTwin task_set '{task_set}'. Choose one of: {choices}")

        # Future frame (future-feature prediction): offset defaults to chunk_size
        self.use_future_feat = use_future_feat
        self.future_frame_offset = future_frame_offset if future_frame_offset is not None else len(indices_config['action_indices'])

        self.state_offsets = torch.tensor(indices_config['state_indices'], dtype=torch.long)
        self.action_offsets = torch.tensor(indices_config['action_indices'], dtype=torch.long)
        self.chunk_size = len(self.action_offsets)
        self.indices_config = indices_config
        self.camera_names = camera_names
        # image_size: (W, H)
        self.image_size = tuple(image_size)
        assert self.image_size[0] % 16 == 0 and self.image_size[1] % 16 == 0, \
            f"DINOv3 requires H/W to be multiples of 16, got {self.image_size}"
        self.val = val
        self.image_aug = image_aug

        # LAVA supervision is training-only. Its temporal convention assumes
        # action token k corresponds to the frame at anchor+k.
        self.use_lava = bool(use_lava) and not val
        self.lava_scales = tuple(int(scale) for scale in (lava_scales or [1, 2, 4, 8, 16]))
        self.lava_sample_ratio = float(lava_sample_ratio)
        self.lava_scale_sampling = str(lava_scale_sampling)
        self.lava_scale_probs = None if lava_scale_probs is None else tuple(float(p) for p in lava_scale_probs)
        self.lava_sampling_balance = str(lava_sampling_balance).lower()
        if self.use_lava:
            if not 0.0 <= self.lava_sample_ratio <= 1.0:
                raise ValueError(f"lava_sample_ratio must be in [0,1], got {self.lava_sample_ratio}")
            if not self.lava_scales or any(scale < 1 or scale >= self.chunk_size for scale in self.lava_scales):
                raise ValueError(
                    f"Every LAVA scale must satisfy 1 <= scale < action chunk size {self.chunk_size}; "
                    f"got {self.lava_scales}")
            if not torch.equal(self.action_offsets, torch.arange(self.chunk_size)):
                raise ValueError(
                    "LAVA requires contiguous action_indices [0, ..., chunk_size-1] for temporal alignment")
            if self.lava_scale_sampling not in {"uniform", "weighted"}:
                raise ValueError(
                    f"lava_scale_sampling must be 'uniform' or 'weighted', got {self.lava_scale_sampling}")
            if self.lava_scale_sampling == "weighted":
                if self.lava_scale_probs is None or len(self.lava_scale_probs) != len(self.lava_scales):
                    raise ValueError("weighted LAVA scale sampling requires one probability per scale")
                if any(p < 0 for p in self.lava_scale_probs) or sum(self.lava_scale_probs) <= 0:
                    raise ValueError("lava_scale_probs must be non-negative with a positive sum")
            if self.lava_sampling_balance not in {"none", "task", "task_episode"}:
                raise ValueError(
                    "lava_sampling_balance must be one of 'none', 'task', or "
                    f"'task_episode', got {self.lava_sampling_balance}")

        # Task condition vector directory
        self.task_cond_dir = task_cond_dir
        self.task_cond_cache = {}   # task_name -> torch.Tensor(D,)
        self.use_task_cond = task_cond_dir is not None

        # ColorJitter augmentation pool
        self.aug_pool = []
        if self.image_aug and not self.val:
            logger.info("Initializing Image Augmentation (Randomly picking 1-2 ops)...")
            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05),
            ]

        # 1. Scan all episodes
        self._load_episodes()

        if self.use_task_cond:
            self._load_task_cond_vectors()

        # 2. Build the flat index map
        logger.info("Building Index Map for dataset...")
        self._build_index_map()
        self._build_lava_sampling_probabilities()

        if self.use_future_feat:
            logger.info(f"Future-Feat mode ENABLED: will return future frame at offset={self.future_frame_offset}.")
        if self.use_lava:
            logger.info(
                f"LAVA sampling ENABLED: scales={list(self.lava_scales)}, "
                f"sample_ratio={self.lava_sample_ratio}, scale_sampling={self.lava_scale_sampling}, "
                f"balance={self.lava_sampling_balance}.")

    @staticmethod
    def _equalized_expected_counts(capacities, requested_total):
        """Water-fill a target while keeping unsaturated groups equally represented."""
        capacities = np.asarray(capacities, dtype=np.float64)
        if capacities.ndim != 1 or np.any(capacities < 0):
            raise ValueError("capacities must be a one-dimensional non-negative array")
        target = min(max(float(requested_total), 0.0), float(capacities.sum()))
        allocation = np.zeros_like(capacities)
        active = set(np.flatnonzero(capacities > 0).tolist())
        remaining = target
        while active and remaining > 0:
            share = remaining / len(active)
            saturated = [index for index in active if capacities[index] <= share]
            if not saturated:
                for index in active:
                    allocation[index] = share
                remaining = 0.0
                break
            for index in saturated:
                allocation[index] = capacities[index]
                remaining -= capacities[index]
                active.remove(index)
        return allocation

    def _build_lava_sampling_probabilities(self):
        """Precompute per-episode probabilities without changing the base loader."""
        episode_count = len(self.episode_metadata)
        self._lava_episode_probabilities = np.full(
            episode_count, self.lava_sample_ratio, dtype=np.float64)
        self.lava_sampling_summary = {}
        if not self.use_lava or self.lava_sampling_balance == "none":
            return

        min_scale = min(self.lava_scales)
        eligible_by_episode = np.asarray([
            max(0, int(meta['length']) - min_scale)
            for meta in self.episode_metadata
        ], dtype=np.float64)
        task_names = sorted({meta['task_name'] for meta in self.episode_metadata})
        episode_indices_by_task = {
            task_name: np.asarray([
                index for index, meta in enumerate(self.episode_metadata)
                if meta['task_name'] == task_name
            ], dtype=np.int64)
            for task_name in task_names
        }
        task_capacities = np.asarray([
            eligible_by_episode[episode_indices_by_task[task_name]].sum()
            for task_name in task_names
        ], dtype=np.float64)
        requested_total = self.lava_sample_ratio * float(eligible_by_episode.sum())
        task_targets = self._equalized_expected_counts(task_capacities, requested_total)

        probabilities = np.zeros(episode_count, dtype=np.float64)
        for task_index, task_name in enumerate(task_names):
            episode_indices = episode_indices_by_task[task_name]
            episode_capacities = eligible_by_episode[episode_indices]
            task_target = float(task_targets[task_index])
            if self.lava_sampling_balance == "task":
                task_capacity = float(episode_capacities.sum())
                episode_targets = episode_capacities * (
                    task_target / task_capacity if task_capacity > 0 else 0.0)
            else:
                episode_targets = self._equalized_expected_counts(
                    episode_capacities, task_target)
            episode_probabilities = np.divide(
                episode_targets,
                episode_capacities,
                out=np.zeros_like(episode_targets),
                where=episode_capacities > 0,
            )
            probabilities[episode_indices] = episode_probabilities
            self.lava_sampling_summary[task_name] = {
                'episodes': int(len(episode_indices)),
                'eligible_anchors': int(episode_capacities.sum()),
                'expected_lava_samples': float(episode_targets.sum()),
                'min_episode_probability': float(episode_probabilities.min()),
                'max_episode_probability': float(episode_probabilities.max()),
            }

        if np.any(probabilities < 0) or np.any(probabilities > 1 + 1e-9):
            raise RuntimeError("Invalid task-balanced LAVA sampling probability")
        self._lava_episode_probabilities = np.clip(probabilities, 0.0, 1.0)
        logger.info(
            "LAVA sampling balance=%s targets %.1f samples/epoch across %d tasks "
            "(%.1f total).",
            self.lava_sampling_balance,
            float(task_targets.max()) if len(task_targets) else 0.0,
            len(task_names),
            float(task_targets.sum()),
        )
        for task_name in task_names:
            summary = self.lava_sampling_summary[task_name]
            logger.info(
                "  LAVA task=%s episodes=%d eligible=%d expected=%.1f p_episode=[%.4f, %.4f]",
                task_name,
                summary['episodes'],
                summary['eligible_anchors'],
                summary['expected_lava_samples'],
                summary['min_episode_probability'],
                summary['max_episode_probability'],
            )

    def _sample_lava_interval(self, local_anchor_idx, total_frames, episode_idx=None):
        """Return (start, scale) fully contained in both episode and action chunk."""
        if not self.use_lava:
            return None
        sample_probability = self.lava_sample_ratio
        if self.lava_sampling_balance != "none":
            if episode_idx is None:
                raise ValueError("episode_idx is required for balanced LAVA sampling")
            sample_probability = float(self._lava_episode_probabilities[episode_idx])
        if random.random() >= sample_probability:
            return None
        max_transition = min(self.chunk_size - 1, total_frames - 1 - local_anchor_idx)
        available = [scale for scale in self.lava_scales if scale <= max_transition]
        if not available:
            return None

        if self.lava_scale_sampling == "uniform":
            scale = random.choice(available)
        else:
            probability_by_scale = dict(zip(self.lava_scales, self.lava_scale_probs))
            weights = [probability_by_scale[scale] for scale in available]
            scale = random.choices(available, weights=weights, k=1)[0]
        start = random.randint(0, max_transition - scale)
        return start, scale

    def _build_index_map(self):
        """Iterate over all episodes, collect metadata and build the flat index"""
        self.valid_indices = []
        self.episode_metadata = []

        current_offset = 0
        valid_ep_count = 0

        for ep_info in tqdm(self.all_episodes, desc="Scanning episode lengths"):
            path = ep_info['hdf5_path']
            with h5py.File(path, 'r') as f:
                length = f['joint_action']['vector'].shape[0]

            if length < 2:
                continue

            self.episode_metadata.append({
                'hdf5_path': path,
                'task_name': ep_info['task_name'],
                'split': ep_info.get('split'),
                'length': length,
                'global_start': current_offset,
                'global_end': current_offset + length,
            })

            ep_start = current_offset
            ep_end = current_offset + length
            curr_indices = np.arange(ep_start, ep_end, dtype=np.int64)
            self.valid_indices.append(curr_indices)

            current_offset += length
            valid_ep_count += 1

        self.valid_indices = np.concatenate(self.valid_indices)
        self._ep_end_bounds = np.array([ep['global_end'] for ep in self.episode_metadata])

        logger.info(f"Index map built: {valid_ep_count} valid episodes, {len(self.valid_indices)} total searchable frames.")

    def _load_task_cond_vectors(self):
        """Load precomputed task condition vectors {task_cond_dir}/{task_name}/task_cond.npy"""
        cond_root = Path(self.task_cond_dir)
        if not cond_root.exists():
            raise FileNotFoundError(f"task_cond_dir not found: {cond_root}")

        # Collect all task names present in the dataset
        task_names = set(ep['task_name'] for ep in self.all_episodes)
        loaded = 0
        for tn in sorted(task_names):
            npy_path = cond_root / tn / "task_cond.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Task condition vector missing for task '{tn}': {npy_path}\n"
                    f"Please run precompute_task_cond.py first to generate it.")
            vec = np.load(npy_path).astype(np.float32)
            self.task_cond_cache[tn] = torch.from_numpy(vec)
            loaded += 1
        dim = next(iter(self.task_cond_cache.values())).shape[0]
        logger.info(f"Loaded task condition vectors for {loaded} tasks (dim={dim}) from {cond_root}")

    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        data_dir = task_path / "data"
        if not data_dir.exists():
            return []
        valid_episodes = []
        hdf5_paths = sorted(
            data_dir.glob("*.hdf5"),
            key=lambda p: int(p.stem.removeprefix("episode")),
        )
        for hdf5_path in hdf5_paths:
            valid_episodes.append({
                'episode_name': hdf5_path.stem,
                'task_name': task_path.parent.name if task_path.name in ['demo_clean', 'demo_randomized'] else task_path.name,
                'hdf5_path': str(hdf5_path),
                'split': split_name,
            })
        return valid_episodes

    def _load_episodes(self):
        """Walk the given root directories and find all task folders"""
        scope = "all 50 tasks" if self.selected_tasks is None else f"{self.task_set} ({len(self.selected_tasks)} tasks)"
        logger.info(f"Scanning dataset folders for task set: {scope}")
        data_splits = ["demo_clean", "demo_randomized"] if self.data_mode == "both" else [f"demo_{self.data_mode}"]
        found_tasks = set()

        for root_dir in self.dataset_dirs:
            if not root_dir.exists():
                continue
            for task_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
                if self.selected_tasks is not None and task_dir.name not in self.selected_tasks:
                    continue
                found_tasks.add(task_dir.name)
                task_episodes = []
                for split in data_splits:
                    split_path = task_dir / split
                    if split_path.exists():
                        episodes = self._scan_task_folder(split_path, split)
                        task_episodes.extend(episodes)

                self.all_episodes.extend(task_episodes)

        if self.selected_tasks is not None:
            missing_tasks = sorted(self.selected_tasks - found_tasks)
            if missing_tasks:
                raise FileNotFoundError(
                    f"RoboTwin task_set '{self.task_set}' is missing task directories: {missing_tasks}")

        if not self.all_episodes:
            raise ValueError(f"No valid episodes found in: {self.dataset_dirs}")

        logger.info(
            f"Successfully scanned {len(self.all_episodes)} episode files "
            f"from {len(found_tasks)} tasks.")

    def _get_query_indices(self, query_idx: int, episode_len: int) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
        ep_start, ep_end = 0, episode_len
        query_indices, padding_mask = {}, {}
        keys_to_process = {
            'state': self.indices_config['state_indices'],
            'action': self.indices_config['action_indices'],
        }
        for cam_name in self.camera_names:
            keys_to_process[cam_name] = self.indices_config['camera_indices']

        for key, delta_list in keys_to_process.items():
            abs_indices = [query_idx + delta for delta in delta_list]
            query_indices[key] = [max(ep_start, min(ep_end - 1, idx)) for idx in abs_indices]

            if key == 'action':
                valid_mask = [(idx < ep_end) for idx in abs_indices]
                padding_mask[f"{key}_mask"] = torch.from_numpy(np.array(valid_mask, dtype=bool))
            else:
                padding_mask[f"{key}_mask"] = torch.ones(len(abs_indices), dtype=torch.bool)

        return query_indices, padding_mask

    def _load_hdf5_data(self, hdf5_path: str, query_indices: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        with h5py.File(hdf5_path, 'r') as root:
            # Actions
            t_idx = np.array(query_indices['action'])
            h5_idx = np.unique(t_idx)
            data_batch['action_sequence'] = torch.from_numpy(
                root['joint_action']['vector'][h5_idx][np.searchsorted(h5_idx, t_idx)]
            ).float()

            # State (endpose)
            t_idx = np.array(query_indices['state'])
            h5_idx = np.unique(t_idx)
            l_pose = root['endpose']['left_endpose'][h5_idx]
            l_grip = root['endpose']['left_gripper'][h5_idx]
            r_pose = root['endpose']['right_endpose'][h5_idx]
            r_grip = root['endpose']['right_gripper'][h5_idx]
            if l_grip.ndim == 1:
                l_grip = l_grip[:, None]
            if r_grip.ndim == 1:
                r_grip = r_grip[:, None]
            state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            data_batch['state'] = torch.from_numpy(state_data[np.searchsorted(h5_idx, t_idx)]).float()

            # Cameras
            data_batch['frame'] = {}
            for cam in self.camera_names:
                t_idx = np.array(query_indices[cam])
                if cam not in root['observation']:
                    continue
                h5_idx = np.unique(t_idx)
                comp_imgs = root['observation'][cam]['rgb'][h5_idx]

                decoded = np.stack([_decode(img) for img in comp_imgs])
                final_img = np.ascontiguousarray(decoded[np.searchsorted(h5_idx, t_idx)])
                if (final_img.shape[2], final_img.shape[1]) != self.image_size:
                    final_img = np.stack([
                        cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in final_img
                    ])
                data_batch['frame'][cam] = final_img

            # Future frame (future-feature prediction): single frame, primary camera only
            if self.use_future_feat and 'future_frame' in query_indices:
                cam0 = self.camera_names[0]
                if cam0 in root['observation']:
                    t_idx_f = np.array(query_indices['future_frame'])
                    h5_idx_f = np.unique(t_idx_f)
                    comp_f = root['observation'][cam0]['rgb'][h5_idx_f]
                    decoded_f = np.stack([_decode(img) for img in comp_f])
                    fut_img = np.ascontiguousarray(decoded_f[np.searchsorted(h5_idx_f, t_idx_f)])
                    if (fut_img.shape[2], fut_img.shape[1]) != self.image_size:
                        fut_img = np.stack([
                            cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in fut_img
                        ])
                    data_batch['future_frame'] = fut_img   # (1, H, W, 3) uint8 RGB

            # LAVA evolution frames: primary camera, no color augmentation.
            if self.use_lava and 'evolution_frames' in query_indices:
                cam0 = self.camera_names[0]
                if cam0 in root['observation']:
                    t_idx_e = np.array(query_indices['evolution_frames'])
                    h5_idx_e = np.unique(t_idx_e)
                    comp_e = root['observation'][cam0]['rgb'][h5_idx_e]
                    decoded_e = np.stack([_decode(img) for img in comp_e])
                    evolution_img = np.ascontiguousarray(
                        decoded_e[np.searchsorted(h5_idx_e, t_idx_e)])
                    if (evolution_img.shape[2], evolution_img.shape[1]) != self.image_size:
                        evolution_img = np.stack([
                            cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR)
                            for i in evolution_img
                        ])
                    data_batch['evolution_frames'] = evolution_img
        return data_batch

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        global_curr_idx = self.valid_indices[idx]
        ep_idx = np.searchsorted(self._ep_end_bounds, global_curr_idx, side='right')
        ep_meta = self.episode_metadata[ep_idx]

        abs_start = ep_meta['global_start']

        try:
            # Local index within the episode
            local_anchor_idx = global_curr_idx - abs_start
            total_frames = ep_meta['length']

            query_indices, padding_mask = self._get_query_indices(local_anchor_idx, total_frames)

            lava_interval = self._sample_lava_interval(
                local_anchor_idx, total_frames, episode_idx=ep_idx)
            if lava_interval is not None:
                lava_start, lava_scale = lava_interval
                query_indices['evolution_frames'] = list(range(
                    local_anchor_idx + lava_start,
                    local_anchor_idx + lava_start + lava_scale + 1,
                ))

            # Future frame query index (single frame, anchor + offset)
            if self.use_future_feat:
                future_abs_idx = local_anchor_idx + self.future_frame_offset
                query_indices['future_frame'] = [
                    max(0, min(total_frames - 1, future_abs_idx))
                ]

            data_batch = self._load_hdf5_data(ep_meta['hdf5_path'], query_indices)
            data_batch.update(padding_mask)

            # Primary camera -> ImageNet-normalized pixel_values
            primary_cam = self.camera_names[0]
            pixel_values = None
            if primary_cam in data_batch['frame']:
                imgs_np = data_batch['frame'][primary_cam]   # (T, H, W, 3) uint8 RGB, T=1 for camera_indices=[0]

                # Color augmentation (PIL-based)
                if self.aug_pool:
                    num_ops = random.choice([1, 2])
                    active_ops = random.sample(self.aug_pool, num_ops)
                    new_imgs = []
                    for img_np in imgs_np:
                        pil_img = Image.fromarray(img_np)
                        for op in active_ops:
                            pil_img = op(pil_img)
                        new_imgs.append(np.array(pil_img))
                    imgs_np = np.stack(new_imgs, axis=0)

                # Normalize
                normed = np.stack([_normalize_image(img) for img in imgs_np], axis=0)   # (T, 3, H, W)
                pixel_values = torch.from_numpy(normed).float()
                # camera_indices defaults to [0] -> T=1, squeeze directly to (3, H, W)
                if pixel_values.shape[0] == 1:
                    pixel_values = pixel_values.squeeze(0)

            result = {
                'state': data_batch['state'],
                'action_sequence': data_batch['action_sequence'],
                'pixel_values': pixel_values,
                'state_mask': data_batch['state_mask'],
                'action_mask': data_batch['action_mask'],
                # Diagnostic metadata only; it is never used as a model input.
                'task_name': ep_meta['task_name'],
            }

            # Future-frame pixel_values (no color augmentation; keep the supervision target clean)
            if self.use_future_feat and 'future_frame' in data_batch:
                fut_np = data_batch['future_frame']                        # (1, H, W, 3)
                fut_normed = np.stack([_normalize_image(img) for img in fut_np], axis=0)  # (1, 3, H, W)
                future_pixel_values = torch.from_numpy(fut_normed).float()
                if future_pixel_values.shape[0] == 1:
                    future_pixel_values = future_pixel_values.squeeze(0)   # (3, H, W)
                result['future_pixel_values'] = future_pixel_values

            if lava_interval is not None and 'evolution_frames' in data_batch:
                evolution_np = data_batch['evolution_frames']
                evolution_normed = np.stack(
                    [_normalize_image(img) for img in evolution_np], axis=0)
                result['evolution_pixel_values'] = torch.from_numpy(evolution_normed).float()
                result['evolution_start'] = lava_start
                result['evolution_scale'] = lava_scale

            # Task condition vector (looked up by the episode's task_name)
            if self.use_task_cond:
                task_name = ep_meta['task_name']
                result['task_cond'] = self.task_cond_cache[task_name]   # (D,)

            return result

        except Exception as e:
            logger.warning(f"Error loading idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataset(config: Any, val: bool = False):
    """
    Factory function: create a RobotWinTaskDataset instance (DINOv3 version)
    """
    if config.dataset.get('type', 'robotwin').lower() == 'libero':
        from .libero_dataset import create_libero_dataset
        return create_libero_dataset(config, val=val)

    from omegaconf import OmegaConf
    indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)

    image_size = tuple(OmegaConf.to_container(config.dataset.image_size, resolve=True))

    # Task condition vector directory (optional)
    task_cond_dir = None
    if config.model.get('use_task_cond', False):
        task_cond_dir = config.dataset.get('task_cond_dir', None)
        if task_cond_dir is None:
            raise ValueError("model.use_task_cond=True but dataset.task_cond_dir is not set")

    # Future feature prediction (optional): whether to fetch a future frame
    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    future_frame_offset = config.dataset.get('future_frame_offset', None)

    lava_cfg = config.model.get('lava', {})
    use_lava = bool(lava_cfg.get('enabled', False)) if lava_cfg else False
    lava_scales = list(config.training.get('lava_scales', [1, 2, 4, 8, 16]))
    lava_sample_ratio = float(config.training.get('lava_sample_ratio', 0.25))
    lava_scale_sampling = str(config.training.get('lava_scale_sampling', 'uniform'))
    lava_sampling_balance = str(config.training.get('lava_sampling_balance', 'none'))
    lava_scale_probs = config.training.get('lava_scale_probs', None)
    if lava_scale_probs is not None:
        lava_scale_probs = list(lava_scale_probs)

    params = {
        'dataset_dir': config.dataset.dataset_dir,
        'indices_config': indices_config,
        'val': val,
        'image_aug': config.dataset.image_aug and not val,
        'camera_names': list(config.dataset.camera_names),
        'data_mode': config.dataset.data_mode,
        'image_size': image_size,
        'task_cond_dir': task_cond_dir,
        'use_future_feat': use_future_feat,
        'future_frame_offset': future_frame_offset,
        'task_set': config.dataset.get('task_set', '50'),
        'use_lava': use_lava,
        'lava_scales': lava_scales,
        'lava_sample_ratio': lava_sample_ratio,
        'lava_scale_sampling': lava_scale_sampling,
        'lava_scale_probs': lava_scale_probs,
        'lava_sampling_balance': lava_sampling_balance,
    }

    return RobotWinTaskDataset(**params)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    result = {}
    lava_keys = {'evolution_pixel_values', 'evolution_start', 'evolution_scale'}
    keys = [key for key in batch[0].keys() if key not in lava_keys]

    for key in keys:
        val = batch[0][key]
        if isinstance(val, torch.Tensor):
            result[key] = torch.stack([sample[key] for sample in batch])
        elif val is None:
            result[key] = None
        else:
            # Skip unknown types
            result[key] = [sample[key] for sample in batch]

    lava_samples = [
        (batch_idx, sample) for batch_idx, sample in enumerate(batch)
        if sample.get('evolution_pixel_values') is not None
    ]
    if lava_samples:
        result['evolution_pixel_values'] = [
            sample['evolution_pixel_values'] for _, sample in lava_samples]
        result['evolution_batch_indices'] = torch.tensor(
            [batch_idx for batch_idx, _ in lava_samples], dtype=torch.long)
        result['evolution_starts'] = torch.tensor(
            [sample['evolution_start'] for _, sample in lava_samples], dtype=torch.long)
        result['evolution_scales'] = torch.tensor(
            [sample['evolution_scale'] for _, sample in lava_samples], dtype=torch.long)
    else:
        result['evolution_pixel_values'] = None
        result['evolution_batch_indices'] = None
        result['evolution_starts'] = None
        result['evolution_scales'] = None

    return result

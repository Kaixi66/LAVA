"""LIBERO HDF5 dataset adapter for LiLa-WAM training.

The official LIBERO release stores one HDF5 file per task and multiple
``data/demo_*`` groups per file.  This loader emits the same batch keys as the
RoboTwin loader while preserving both the agent and wrist camera views.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm


logger = logging.getLogger(__name__)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(image, (2, 0, 1))


def task_name_from_path(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_demo") else name


def _read_rows(dataset, indices: List[int]) -> np.ndarray:
    """Read possibly repeated/padded HDF5 rows while satisfying h5py ordering rules."""
    requested = np.asarray(indices, dtype=np.int64)
    unique = np.unique(requested)
    return np.asarray(dataset[unique][np.searchsorted(unique, requested)])


class LiberoTaskDataset(data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        indices_config: Dict[str, List[int]],
        camera_names: List[str],
        image_size=(256, 256),
        val: bool = False,
        image_aug: bool = False,
        task_cond_dir: Optional[str] = None,
        use_future_feat: bool = False,
        future_frame_offset: Optional[int] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"LIBERO dataset directory not found: {self.dataset_dir}")
        self.indices_config = indices_config
        self.state_offsets = list(indices_config["state_indices"])
        self.action_offsets = list(indices_config["action_indices"])
        self.camera_offsets = list(indices_config.get("camera_indices", [0]))
        if self.camera_offsets != [0]:
            raise ValueError("LIBERO currently expects camera_indices: [0]")
        self.camera_names = list(camera_names)
        self.image_size = tuple(image_size)
        if self.image_size[0] % 16 or self.image_size[1] % 16:
            raise ValueError(f"DINOv3 requires image dimensions divisible by 16, got {self.image_size}")
        self.val = val
        self.use_future_feat = use_future_feat
        self.future_frame_offset = (
            future_frame_offset if future_frame_offset is not None else len(self.action_offsets)
        )

        self.aug_pool = []
        if image_aug and not val:
            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05),
            ]

        self.task_cond_dir = Path(task_cond_dir) if task_cond_dir else None
        self.task_cond_cache: Dict[str, torch.Tensor] = {}
        self.episodes = []
        self.index = []
        self._scan()
        if self.task_cond_dir is not None:
            self._load_task_conditions()

    def _scan(self):
        files = sorted(self.dataset_dir.glob("*.hdf5"))
        if not files:
            files = sorted(self.dataset_dir.rglob("*.hdf5"))
        for hdf5_path in tqdm(files, desc="Scanning LIBERO episodes"):
            task_name = task_name_from_path(hdf5_path)
            with h5py.File(hdf5_path, "r") as root:
                if "data" not in root:
                    continue
                for demo_name in sorted(root["data"].keys()):
                    demo = root["data"][demo_name]
                    length = int(demo["actions"].shape[0])
                    if length < 2:
                        continue
                    episode_idx = len(self.episodes)
                    self.episodes.append(
                        {
                            "hdf5_path": str(hdf5_path),
                            "demo_name": demo_name,
                            "task_name": task_name,
                            "length": length,
                        }
                    )
                    self.index.extend((episode_idx, t) for t in range(length))
        if not self.episodes:
            raise ValueError(f"No LIBERO demonstrations found under {self.dataset_dir}")
        logger.info(
            "LIBERO scan complete: %d tasks/files, %d episodes, %d frames",
            len(files), len(self.episodes), len(self.index),
        )

    def _load_task_conditions(self):
        if not self.task_cond_dir.exists():
            raise FileNotFoundError(f"task_cond_dir not found: {self.task_cond_dir}")
        for task_name in sorted({ep["task_name"] for ep in self.episodes}):
            path = self.task_cond_dir / task_name / "task_cond.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing VTT for LIBERO task '{task_name}': {path}. "
                    "Run Kaixi_scripts/prepare_libero.sh first."
                )
            self.task_cond_cache[task_name] = torch.from_numpy(np.load(path).astype(np.float32))

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _bounded_indices(anchor: int, offsets: List[int], length: int):
        return [max(0, min(length - 1, anchor + offset)) for offset in offsets]

    def _prepare_image(self, image: np.ndarray, augment: bool) -> torch.Tensor:
        if (image.shape[1], image.shape[0]) != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        if augment and self.aug_pool:
            for op in random.sample(self.aug_pool, random.choice([1, 2])):
                image = np.asarray(op(Image.fromarray(image)))
        return torch.from_numpy(_normalize_image(np.ascontiguousarray(image))).float()

    def __getitem__(self, item: int):
        episode_idx, anchor = self.index[item]
        ep = self.episodes[episode_idx]
        length = ep["length"]
        action_idx = self._bounded_indices(anchor, self.action_offsets, length)
        state_idx = self._bounded_indices(anchor, self.state_offsets, length)
        future_idx = min(length - 1, anchor + self.future_frame_offset)

        try:
            with h5py.File(ep["hdf5_path"], "r") as root:
                demo = root["data"][ep["demo_name"]]
                obs = demo["obs"]
                actions = np.asarray(_read_rows(demo["actions"], action_idx), dtype=np.float32)
                ee = np.asarray(_read_rows(obs["ee_states"], state_idx), dtype=np.float32)
                gripper = np.asarray(_read_rows(obs["gripper_states"], state_idx), dtype=np.float32)
                state = np.concatenate([ee, gripper], axis=-1)
                views = [
                    self._prepare_image(np.asarray(obs[name][anchor]), augment=not self.val)
                    for name in self.camera_names
                ]
                future = None
                if self.use_future_feat:
                    future = self._prepare_image(
                        np.asarray(obs[self.camera_names[0]][future_idx]), augment=False
                    )

            result = {
                "state": torch.from_numpy(state).float(),
                "action_sequence": torch.from_numpy(actions).float(),
                "pixel_values": torch.stack(views, dim=0),
                "state_mask": torch.ones(len(state_idx), dtype=torch.bool),
                "action_mask": torch.tensor(
                    [anchor + offset < length for offset in self.action_offsets], dtype=torch.bool
                ),
            }
            if future is not None:
                result["future_pixel_values"] = future
            if self.task_cond_dir is not None:
                result["task_cond"] = self.task_cond_cache[ep["task_name"]]
            return result
        except Exception as exc:
            logger.warning("Error loading LIBERO item %d: %s", item, exc)
            return self.__getitem__(random.randrange(len(self)))


def create_libero_dataset(config: Any, val: bool = False):
    indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)
    image_size = tuple(OmegaConf.to_container(config.dataset.image_size, resolve=True))
    task_cond_dir = None
    if config.model.get("use_task_cond", False):
        task_cond_dir = config.dataset.get("task_cond_dir", None)
        if not task_cond_dir:
            raise ValueError("model.use_task_cond=True but dataset.task_cond_dir is not set")
    ff_cfg = config.model.get("future_feat", {})
    return LiberoTaskDataset(
        dataset_dir=config.dataset.dataset_dir,
        indices_config=indices_config,
        camera_names=list(config.dataset.camera_names),
        image_size=image_size,
        val=val,
        image_aug=bool(config.dataset.image_aug and not val),
        task_cond_dir=task_cond_dir,
        use_future_feat=bool(ff_cfg.get("enabled", False)) if ff_cfg else False,
        future_frame_offset=config.dataset.get("future_frame_offset", None),
    )

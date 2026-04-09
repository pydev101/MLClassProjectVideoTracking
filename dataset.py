"""
CSRNet training dataset.

This module provides a PyTorch ``Dataset`` that pairs each crowd image with its
corresponding density-map target.  It supports two ground-truth formats:

  1. **HDF5** (CSRNet-pytorch convention) – ``<stem>.h5`` with a ``density``
     dataset, stored under a ``ground_truth/`` folder parallel to ``images/``.
  2. **MATLAB** (ShanghaiTech convention) – ``GT_<stem>.mat`` containing head
     annotations, converted on-the-fly to a Gaussian density map by
     ``csrnet.load_ground_truth``.

Image paths are supplied either as a JSON file (list of strings or list of
``{"img": ...}`` dicts) or directly as a Python sequence.

During training the density map is:
  - optionally augmented with a random horizontal flip (20 % chance),
  - down-sampled by 8x (matching the CSRNet network stride after 3 max-pools),
  - scaled by 64 so that per-pixel values remain in a numerically convenient
    range (standard CSRNet-pytorch recipe).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Sequence, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from csrnet import load_ground_truth


def _normalize_train_list(root: Union[str, Sequence[Any]]) -> List[str]:
    """Accept a JSON filepath *or* an in-memory list and return a flat list of
    image-path strings.  Handles both ``["path", ...]`` and
    ``[{"img": "path"}, ...]`` formats.
    """
    if isinstance(root, str) and root.endswith(".json"):
        with open(root, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _normalize_train_list(data)
    items = list(root)
    if not items:
        return []
    if isinstance(items[0], dict):
        return [str(it["img"]) for it in items]
    return [str(x) for x in items]


def _density_downsample(target: np.ndarray) -> np.ndarray:
    """Shrink a full-resolution density map to 1/8 its spatial size and
    multiply by 64.

    CSRNet's frontend contains 3 max-pool layers, so the network output is 8x
    smaller than the input.  Multiplying by 8**2 = 64 compensates for the area
    reduction so that the *sum* of the down-sampled map still equals the
    original head count (energy-preserving rescale).
    """
    h, w = target.shape[0], target.shape[1]
    th, tw = h // 8, w // 8
    if th < 1 or tw < 1:
        raise ValueError(f"Image too small for CSRNet downsampling: got map shape {(h, w)}")
    pil = Image.fromarray(np.asarray(target, dtype=np.float32))
    resized = np.array(pil.resize((tw, th), Image.BICUBIC), dtype=np.float32)
    return resized * 64.0


def load_data(img_path: str, train: bool = True) -> tuple[Image.Image, np.ndarray]:
    """
    Load an RGB image and its matching density-map target.

    Ground-truth resolution order:
      1. Look for an ``.h5`` file (CSRNet-pytorch pre-computed density).
      2. Fall back to ``GT_<stem>.mat`` (ShanghaiTech head annotations ->
         Gaussian density built on the fly).

    When ``train=True``, a random horizontal flip is applied 20 % of the time
    as a simple data-augmentation step.  The density map is always down-sampled
    to 1/8 resolution (see ``_density_downsample``).
    """
    img_path = str(img_path)
    img = Image.open(img_path).convert("RGB")
    p = Path(img_path)

    # Convention: ground truth lives in a sibling "ground_truth" folder with
    # the same stem as the image.
    h5_path = str(p.with_suffix(".h5")).replace("images", "ground_truth")
    mat_path = p.parent.parent / "ground_truth" / f"GT_{p.stem}.mat"

    if Path(h5_path).is_file():
        with h5py.File(h5_path, "r") as gt_file:
            target = np.asarray(gt_file["density"], dtype=np.float32)
    elif mat_path.is_file():
        target = load_ground_truth(mat_path, image_path=img_path).astype(np.float32)
    else:
        raise FileNotFoundError(
            f"No ground truth found: tried {h5_path!s} and {mat_path!s} for image {img_path}"
        )

    # Data augmentation: 20 % chance of a left-right flip.
    if train:
        if random.random() > 0.8:
            target = np.ascontiguousarray(np.fliplr(target))
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Reduce density to the network's output resolution before returning.
    target = _density_downsample(target)
    return img, target


class listDataset(Dataset):
    """
    PyTorch Dataset that wraps a list of image paths for CSRNet training /
    evaluation.

    Each ``__getitem__`` call loads the RGB image, retrieves (or generates) the
    density-map target, applies the optional ``transform`` (typically ImageNet
    normalisation), and returns ``(image_tensor, target_tensor)``.

    For training, the path list is repeated 4x (effectively 4 augmented passes
    per epoch) and optionally shuffled.
    """
    def __init__(
        self,
        root: Union[str, Sequence[Any]],
        shape: Any = None,
        shuffle: bool = True,
        transform: Any = None,
        train: bool = False,
        seen: int = 0,
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        """
        Args:
            root: JSON filepath *or* an in-memory sequence of image paths.
            shape: Reserved (unused) -- kept for API compatibility.
            shuffle: Shuffle the path list when ``train=True``.
            transform: torchvision transform applied to each PIL image (e.g.
                ``ToTensor`` + ``Normalize``).
            train: Enables data augmentation (random flip) and 4x list repeat.
            seen: Running sample counter (shared with the model for some
                scheduling strategies).
            batch_size: Informational; the actual batch size is set on the
                ``DataLoader``.
            num_workers: Informational; the actual worker count is set on the
                ``DataLoader``.
        """
        self.shape = shape
        self.transform = transform
        self.train = train
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        lines = _normalize_train_list(root)
        if train:
            # Repeat the dataset 4x so each epoch sees more augmented variants.
            lines = lines * 4
            if shuffle:
                random.shuffle(lines)

        self.lines = lines
        self.nSamples = len(lines)

    def __len__(self) -> int:
        return self.nSamples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(image_tensor, target_tensor)`` for the given index.

        ``image_tensor`` has shape ``[3, H, W]`` (after ``transform``).
        ``target_tensor`` has shape ``[1, H/8, W/8]`` (single-channel density).
        """
        if index < 0 or index >= len(self):
            raise IndexError("index range error")
        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)
        if self.transform is not None:
            img = self.transform(img)
        # Wrap the 2D numpy density in a [1, H, W] tensor (batch-channel dim).
        target_t = torch.from_numpy(target).float().unsqueeze(0)
        return img, target_t

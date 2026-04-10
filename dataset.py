"""
CSRNet training dataset.

This module provides a PyTorch ``Dataset`` that pairs each crowd image with its
corresponding density-map target.  It supports three ground-truth formats:

  1. **HDF5** (CSRNet-pytorch convention) – ``<stem>.h5`` with a ``density``
     dataset, stored under a ``ground_truth/`` folder parallel to ``images/``.
  2. **MATLAB** (ShanghaiTech convention) – ``GT_<stem>.mat`` containing head
     annotations, converted on-the-fly to a Gaussian density map by
     ``load_ground_truth``.
  3. **MATLAB** (Mall convention) – a single ``mall_gt.mat`` next to the
     ``frames/`` directory, with images named ``seq_NNNNNN.jpg``; uses
     ``csrnet.mall_load_ground_truth_density_map`` (with an in-memory cache so
     the ``.mat`` file is not re-read every frame).

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
import scipy.io as sio
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.io.matlab._mio5_params import mat_struct
from torch.utils.data import Dataset
from typing import Optional, Tuple

from csrnet import mall_frame_index_from_image_path, mall_load_ground_truth_density_map

# One loaded dict per ``mall_gt.mat`` path (large file; shared across all frames).
_mall_mat_cache: dict[str, dict] = {}


def _get_mall_mat_dict(gt_path: Path) -> dict:
    key = str(gt_path.resolve())
    if key not in _mall_mat_cache:
        _mall_mat_cache[key] = sio.loadmat(gt_path, squeeze_me=False, struct_as_record=False)
    return _mall_mat_cache[key]


def load_ground_truth(
    ground_truth_path: Union[str, Path],
    *,
    image_path: Optional[Union[str, Path]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    density_sigma: float = 4.0,
) -> np.ndarray:
    """Load ShanghaiTech-style ground truth from a ``.mat`` file and build a
    2D density map.

    Each head annotation becomes a unit impulse on an image-sized canvas, then
    an isotropic Gaussian (sigma=``density_sigma``) is applied so each head
    integrates to ~1.0 and the map sum ≈ head count.

    Args:
        ground_truth_path: Path to ``GT_*.mat`` (must contain ``image_info``).
        image_path: RGB image whose dimensions define the map size.
        image_shape: ``(height, width)`` alternative to ``image_path``.
        density_sigma: Gaussian sigma; pass 0 for raw impulses.
    """
    if image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
    elif image_path is not None:
        with Image.open(image_path) as im:
            w, h = im.size
    else:
        raise ValueError("Provide image_shape (H, W) or image_path so the density map size is known.")

    mat = sio.loadmat(Path(ground_truth_path), squeeze_me=False, struct_as_record=False)
    points = _shanghaitech_points_from_mat(mat)

    density = np.zeros((h, w), dtype=np.float64)
    for x, y in points:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if 0 <= yi < h and 0 <= xi < w:
            density[yi, xi] += 1.0

    if density_sigma > 0.0:
        density = gaussian_filter(density, sigma=float(density_sigma))

    return density


def _normalize_gt_points(points: np.ndarray) -> np.ndarray:
    """Ensure head annotations are a float64 array of shape ``(N, 2)`` — (x, y)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[-1] != 2:
        raise ValueError(f"Expected 2 columns (x, y); got shape {pts.shape}")
    return pts.reshape(-1, 2)


def _unwrap_location_array(loc: Any) -> np.ndarray:
    """Peel (1,1) object-cell wrappers until the coordinate array is reached."""
    cur = loc
    for _ in range(8):
        if isinstance(cur, np.ndarray) and cur.shape == (1, 1):
            cur = cur[0, 0]
            continue
        break
    return np.asarray(cur, dtype=np.float64)


def _shanghaitech_points_from_mat(mat: dict) -> np.ndarray:
    """Extract (N, 2) head coordinates from a ShanghaiTech ``GT_*.mat``."""
    if "image_info" not in mat:
        raise KeyError("Expected 'image_info' in .mat (ShanghaiTech GT format).")
    info = mat["image_info"]

    # squeeze_me=True, struct_as_record=True produces a 0-d structured array.
    if isinstance(info, np.ndarray) and info.shape == () and info.dtype.names:
        names = info.dtype.names
        if "location" in names:
            row = dict(zip(names, info.item()))
            return _normalize_gt_points(_unwrap_location_array(row["location"]))
        raise ValueError("Structured image_info has no 'location' field.")

    # Default loadmat: nested (1,1) arrays ending in mat_struct with .location
    cur: Any = info
    for _ in range(10):
        if isinstance(cur, np.ndarray) and cur.shape == (1, 1):
            cur = cur[0, 0]
            continue
        break

    if isinstance(cur, mat_struct):
        if not hasattr(cur, "location"):
            raise ValueError("ShanghaiTech mat_struct has no 'location' field.")
        return _normalize_gt_points(_unwrap_location_array(cur.location))

    # Legacy cell layout fallback.
    try:
        pts = mat["image_info"][0, 0][0, 0][0]
        return _normalize_gt_points(_unwrap_location_array(pts))
    except (TypeError, IndexError, KeyError) as err:
        raise ValueError(
            f"Could not parse ShanghaiTech image_info (type: {type(cur).__name__})."
        ) from err



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
        # Mall: ``.../mall_dataset/mall_gt.mat`` with frames under ``.../frames/seq_*.jpg``
        mall_gt_path = p.parent.parent / "mall_gt.mat"
        if mall_gt_path.is_file():
            try:
                img_index = mall_frame_index_from_image_path(p)
            except ValueError as err:
                raise FileNotFoundError(
                    f"No ground truth found for {img_path}: tried {h5_path!s}, {mat_path!s}, "
                    f"and {mall_gt_path!s} (Mall) but filename is not seq_NNNNNN.jpg ({err})"
                ) from err
            mat_dict = _get_mall_mat_dict(mall_gt_path)
            target = mall_load_ground_truth_density_map(
                mall_gt_path,
                img_index=img_index,
                image_path=img_path,
                mat=mat_dict,
            ).astype(np.float32)
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

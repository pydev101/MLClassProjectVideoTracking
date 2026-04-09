# Author: Taylor Lindley
# Date: 4/4/2026
# Description: Implementation of CSRNet in Python along with utilities for loading
# images, running inference, and parsing ShanghaiTech ground-truth annotations.
#
# CSRNet (Congested Scene Recognition Network) is a CNN architecture for crowd
# density estimation. It takes an arbitrary-size RGB image and produces a 2D
# density map whose pixel-wise sum approximates the total head count.
#
# Architecture overview:
#   Frontend  – First 10 conv layers of VGG-16 (3 max-pools → 1/8 spatial stride).
#   Backend   – 6 dilated 3×3 convolutions that expand the receptive field without
#               further reducing resolution.
#   Output    – A single 1×1 conv that collapses the 64-channel feature map to a
#               1-channel density map.
#
# Reference paper: Li et al., "CSRNet: Dilated Convolutional Neural Networks for
# Understanding the Highly Congested Scenes", CVPR 2018.

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from PIL import Image
from torchvision import transforms
from typing import Union, List, Any, Tuple, Optional, cast
from pathlib import Path
import torch.nn as nn
import torch
from torchvision import models
from scipy.io.matlab._mio5_params import mat_struct


class CSRNet(nn.Module):
    """
    Congested Scene Recognition Network (CSRNet) for crowd density maps.

    The network has two halves:
      1. **Frontend** – a truncated VGG-16 (up to conv3-3, 13 weight layers) that
         extracts multi-scale features while down-sampling by 8× via max-pooling.
      2. **Backend** – six 3×3 dilated convolutions (dilation=2) that widen the
         receptive field *without* further spatial reduction, followed by a 1×1
         conv that produces a single-channel density map.

    During training, the frontend is typically initialised from ImageNet-pretrained
    VGG-16 weights so the model converges faster; the backend and output head are
    trained from scratch.
    """

    def __init__(self, load_weights: bool = False) -> None:
        """
        Construct CSRNet with a VGG-style frontend and dilated backend.
        
        Args:
            load_weights: If **False** (default), download/copy ImageNet VGG-16
                weights into the frontend layers for transfer learning.  
                If **True**, skip pretrained initialisation (used when you plan to
                load a full checkpoint yourself afterwards).
        """
        super(CSRNet, self).__init__()
        # Running count of training samples the model has seen (used by some
        # data-loader logic to coordinate shuffling across epochs).
        self.seen: int = 0
        # Frontend: based on VGG16 architecture (for feature extraction)
        self.frontend_feat: List[Any] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Backend: dilated convolutions (to preserve spatial resolution)
        self.backend_feat: List[Any] = [512, 512, 512, 256, 128, 64]
        
        self.frontend = self.make_layers(self.frontend_feat)
        self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Final output layer reduces channels to a 1D density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            # Transfer-learn: copy the first 10 conv layers (weights + biases)
            # from an ImageNet-pretrained VGG-16 into frontend.
            mod = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            frontend_state_dict = list(self.frontend.state_dict().items())
            mod_state_dict = list(mod.state_dict().items())
            # Iterate over matching parameter slots and overwrite in-place.
            for i in range(len(frontend_state_dict)):
                frontend_state_dict[i][1].data[:] = mod_state_dict[i][1].data[:]


    def make_layers(self, cfg: List[Any], in_channels: int = 3, batch_norm: bool = False, dilation: bool = False) -> nn.Sequential:
        """
        Build a sequential stack of conv / pool / ReLU (and optional BatchNorm) layers from a config list.
        
        Args:
            cfg: List where integers are Conv2d output channels and ``'M'`` marks a 2x2 max-pool.
            in_channels: Input channels for the first conv layer (default 3 for RGB).
            batch_norm: If True, insert BatchNorm after each conv.
            dilation: If True, use dilation 2 on conv layers instead of 1.
            
        Returns:
            nn.Sequential: Layer stack for CSRNet front- or back-end.
        """
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the network: image batch to single-channel density map.
        
        Args:
            x: Input tensor of shape ``[N, 3, H, W]``.
            
        Returns:
            torch.Tensor: Density map of shape ``[N, 1, H', W']`` (spatial size reduced versus input).
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self) -> None:
        """
        Initialize conv and batch-norm parameters (used when wiring VGG16 into the frontend).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


################################################################################################
# Inference & I/O utilities
#
# These helper functions handle the full inference pipeline: transforming a PIL
# image into a normalised batch tensor, loading a saved checkpoint into CSRNet,
# running a forward pass, and up-scaling the small density map back to the
# original image resolution.
################################################################################################

def csrnet_image_transform() -> transforms.Compose:
    """
    Compose ToTensor plus ImageNet mean/std normalization to match VGG-style CSRNet inputs.
    
    Returns:
        transforms.Compose: Preprocessing pipeline for PIL images before batching.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def pil_image_to_csrnet_batch(img: Image.Image, transform: Optional[transforms.Compose] = None) -> torch.Tensor:
    """
    Turn an RGB PIL image into a single-sample batch tensor ready for ``CSRNet.forward``.
    
    Args:
        img: RGB ``PIL.Image``.
        transform: Optional preprocessing; defaults to ``csrnet_image_transform()``.
    
    Returns:
        torch.Tensor: Batch of shape ``[1, 3, H, W]`` in normalized float32.
    """
    if transform is None:
        transform = csrnet_image_transform()
    return cast(torch.Tensor, transform(img).unsqueeze(0))


def load_csrnet_model(weights_path: Union[str, Path], *, map_location: Optional[Union[str, torch.device]] = None) -> CSRNet:
    """
    Instantiate a new ``CSRNet`` and load trained weights from a ``.pth`` checkpoint.
    
    Args:
        weights_path: Filesystem path to the checkpoint.
        map_location: Device string or ``torch.device`` for ``torch.load`` (defaults to CPU).
    
    Returns:
        CSRNet: Model in ``eval`` mode with loaded ``state_dict``.
    """
    # Set the default map location to CPU
    if map_location is None:
        map_location = torch.device("cpu")
    # Create a new CSRNet model
    model = CSRNet()
    # Load the checkpoint
    checkpoint = torch.load(Path(weights_path), map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # Set the model to evaluation mode (since not in training mode here)
    model.eval()
    return model

def csrnet_predict(model: CSRNet, image_batch: torch.Tensor) -> Tuple[np.ndarray, float]:
    """
    Run CSRNet forward pass; return density map ``[H, W]`` and estimated crowd count. Density map is 1/8 of the input image size.
    
    Args:
        model: CSRNet model
        image_batch: Image batch tensor
        
    Returns:
        Tuple[np.ndarray, float]: Density map and estimated crowd count
    """
    # Set the model to evaluation mode b/c not training here
    model.eval()
    # Disable gradient computation to save memory and improve performance (again, not training here)
    with torch.no_grad():
        # Run the model forward pass
        output = model(image_batch)
    # Squeeze the output to remove the batch dimension and convert to numpy
    density_map = output.squeeze().cpu().numpy()
    # Calculate the estimated crowd count by summing the density map
    estimated_count = float(np.sum(density_map))
    # Return the density map and estimated crowd count
    return density_map, estimated_count


def bilinear_interpolation(
    density_map: np.ndarray,
    original_image: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a 2D density map to a target height and width using bilinear interpolation.

    Args:
        density_map: 2D array of shape ``(H_in, W_in)``.
        original_image: Target ``(height, width)`` (same axis order as NumPy row-major maps).
            This matches an RGB image's ``height, width = pil.size[1], pil.size[0]``.

    Returns:
        np.ndarray: Map of shape ``(height, width)``. Floating inputs keep their dtype when
        possible; otherwise values are float32.
    """
    arr = np.asarray(density_map)
    if arr.ndim != 2:
        raise ValueError(f"density_map must be 2D, got shape {arr.shape}")
    th, tw = int(original_image[0]), int(original_image[1])
    if th < 1 or tw < 1:
        raise ValueError("original_image (height, width) entries must be >= 1")

    x = torch.from_numpy(arr.astype(np.float32, copy=False))[None, None, :, :]
    y = torch.nn.functional.interpolate(
        x, size=(th, tw), mode="bilinear", align_corners=False
    )
    out = y.squeeze(0).squeeze(0).numpy()

    if np.issubdtype(arr.dtype, np.floating):
        return out.astype(arr.dtype, copy=False)
    return out


def load_and_process_image_with_csrnet(
        model: CSRNet,
        image_path: Union[str, Path],
        *,
        bilinear_interpolation: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Tuple[Image.Image, np.ndarray, float]:
    """
    Load an image file, preprocess, run CSRNet, and return display image, density map, and count. Density map is 1/8 of the input image size.
    
    Args:
        model: Loaded ``CSRNet`` instance.
        image_path: Path to an image file (RGB read via PIL).
        device: Optional device for model and batch; if omitted, uses the model's current parameter device.
    
    Returns:
        Tuple[Image.Image, np.ndarray, float]: RGB PIL image, 2D density map ``[H, W]``, and sum count.
    """
    pil_img = Image.open(image_path).convert("RGB")
    batch = pil_image_to_csrnet_batch(pil_img)
    if device is not None:
        dev = torch.device(device) if isinstance(device, str) else device
        model.to(dev)
        batch = batch.to(dev)
    else:
        batch = batch.to(next(model.parameters()).device)
    density_map, count = csrnet_predict(model, batch)
    if bilinear_interpolation:
        density_map = bilinear_interpolation(density_map, pil_img.size)
    return pil_img, density_map, count


################################################################################################
# Ground-truth loading (ShanghaiTech .mat format)
#
# ShanghaiTech GT files (GT_IMG_*.mat) store head annotations as (x, y)
# coordinates inside deeply nested MATLAB structs.  The helpers below peel
# through several possible nesting layouts produced by different versions of
# scipy.io.loadmat, extract the Nx2 coordinate array, stamp unit impulses on
# a blank image-sized canvas, and optionally smooth with a Gaussian to produce
# a continuous density map suitable for MSE-based training.
################################################################################################

def _normalize_gt_points(points: np.ndarray) -> np.ndarray:
    """Ensure head annotations are a float64 array of shape ``(N, 2)`` (x, y)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[-1] != 2:
        raise ValueError(f"Expected head coordinates with 2 columns (x, y); got shape {pts.shape}")
    return pts.reshape(-1, 2)

def _unwrap_location_array(loc: Any) -> np.ndarray:
    """Peel ``(1, 1)`` object/cell wrappers until we reach the coordinate array.

    MATLAB cell arrays often appear as shape-(1,1) numpy object arrays after
    loadmat.  This recursively indexes [0,0] up to 8 times until the actual
    numeric coordinate data is reached.
    """
    cur = loc
    for _ in range(8):
        if isinstance(cur, np.ndarray) and cur.shape == (1, 1):
            cur = cur[0, 0]
            continue
        break
    return np.asarray(cur, dtype=np.float64)

def _shanghaitech_points_from_mat(mat: dict) -> np.ndarray:
    """Parse ShanghaiTech-style ``GT_*.mat`` head locations; returns ``(N, 2)`` float array (x, y)."""
    if "image_info" not in mat:
        raise KeyError("Expected 'image_info' in .mat (ShanghaiTech GT format).")
    info = mat["image_info"]

    # loadmat(..., squeeze_me=True, struct_as_record=True): 0-d structured ndarray
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

    # Legacy cell layout (some exports): image_info[0,0][0,0][0] is already Nx2
    try:
        pts = mat["image_info"][0, 0][0, 0][0]
        return _normalize_gt_points(_unwrap_location_array(pts))
    except (TypeError, IndexError, KeyError) as err:
        raise ValueError(
            f"Could not parse ShanghaiTech image_info (type after unwrap: {type(cur).__name__})."
        ) from err

def load_ground_truth_density_map(
        ground_truth_path: Union[str, Path],
        *,
        image_path: Optional[Union[str, Path]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        density_sigma: float = 4.0,
    ) -> np.ndarray:
    """
    Load ShanghaiTech-style ground truth from a ``.mat`` file and build a 2D density map.

    Annotations are head locations; each receives a unit impulse, then an isotropic Gaussian
    with standard deviation ``density_sigma`` is applied (typical for CSRNet-style targets).

    Args:
        ground_truth_path: Path to ``GT_*.mat`` (must contain ``image_info``).
        image_path: Optional RGB image path; height/width define map size (same as the image).
        image_shape: Optional ``(height, width)`` if you do not have ``image_path``.
        density_sigma: Gaussian sigma for smoothing; pass ``0`` to keep unsmoothed impulses only.

    Returns:
        np.ndarray: 2D density map ``float64`` of shape ``(H, W)``, summing to approximately
        the head count for positive ``density_sigma``.

    Raises:
        ValueError: If neither ``image_path`` nor ``image_shape`` is given.
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

    # Build a blank image-sized canvas and stamp a unit impulse at each head
    # coordinate.  After Gaussian smoothing, each impulse integrates to ~1.0,
    # so the map's total sum ≈ the head count.
    density = np.zeros((h, w), dtype=np.float64)
    for x, y in points:
        xi, yi = int(np.floor(x)), int(np.floor(y))
        if 0 <= yi < h and 0 <= xi < w:
            density[yi, xi] += 1.0

    # Isotropic Gaussian smoothing converts the sparse impulse map into a
    # continuous density map
    if density_sigma > 0.0:
        density = gaussian_filter(density, sigma=float(density_sigma))

    return density
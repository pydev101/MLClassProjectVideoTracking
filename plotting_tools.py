# Author: Taylor Lindley
# Date: 4/4/2026

from __future__ import annotations
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def plot_density_overlay(
    image: Image.Image,
    density_map: np.ndarray,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    alpha: float = 0.5,
    cmap=CM.jet,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """
    Show an RGB image with a density heatmap overlaid (heatmap stretched to image bounds).

    Args:
        image: Base ``PIL.Image`` in RGB.
        density_map: 2D array (may be lower resolution than the image; ``extent`` matches the image).
        title: Optional axes title.
        figsize: Matplotlib figure size.
        alpha: Heatmap transparency.
        cmap: Colormap for the density channel.
        show: If True, call ``plt.show()``.
        block: Passed to ``plt.show(block=...)``: if True, wait until the figure window is closed
            (when the GUI backend supports it). Defaults to False (non-blocking).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.imshow(
        density_map,
        cmap=cmap,
        alpha=alpha,
        extent=[0, image.size[0], image.size[1], 0],
    )
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show(block=block)
    return fig


def plot_csrnet_density_map(
    image: Image.Image,
    density_map: np.ndarray,
    estimated_count: float,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    alpha: float = 0.5,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """
    Plot a CSRNet predicted density map over the image.

    Args:
        image: RGB ``PIL.Image``.
        density_map: Model output map ``[H, W]`` (e.g.\ 1/8 of input size; overlaid with correct ``extent``).
        estimated_count: Scalar count (e.g.\ sum of density); used in the default title.
        title: Overrides the default title when set.
        figsize: Matplotlib figure size.
        alpha: Heatmap transparency.
        show: If True, call ``plt.show()``.
        block: Forwarded to ``plot_density_overlay`` / ``plt.show`` (default non-blocking).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    if title is None:
        title = f"CSRNet Crowd Density Map\nEstimated Count: {estimated_count:.2f}"
    return plot_density_overlay(
        image,
        density_map,
        title=title,
        figsize=figsize,
        alpha=alpha,
        show=show,
        block=block,
    )


def plot_ground_truth_density(
    image: Image.Image,
    ground_truth_density: np.ndarray,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    alpha: float = 0.5,
    show: bool = True,
    block: bool = False,
) -> Figure:
    """
    Plot a ground-truth density map (e.g.\ from ``csrnet.load_ground_truth``) over the image.

    Args:
        image: RGB ``PIL.Image`` (same geometry as the map / training labels).
        ground_truth_density: 2D density map ``[H, W]`` (typically full image resolution).
        title: Overrides the default title when set.
        figsize: Matplotlib figure size.
        alpha: Heatmap transparency.
        show: If True, call ``plt.show()``.
        block: Forwarded to ``plot_density_overlay`` / ``plt.show`` (default non-blocking).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    if title is None:
        count = float(np.sum(ground_truth_density))
        title = f"Ground Truth Density Map\nCount (sum): {count:.2f}"
    return plot_density_overlay(
        image,
        ground_truth_density,
        title=title,
        figsize=figsize,
        alpha=alpha,
        show=show,
        block=block,
    )

def gaussian_blur(density_map: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Apply a blur to the density map to smooth out the edges.
    
    Args:
        density_map: Density map to blur
        kernel_size: Tuple of kernel size for the blur
        
    Returns:
        np.ndarray: Blurred density map with the same shape and dtype as the input.
    """
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    if ky < 1 or kx < 1:
        raise ValueError("kernel_size entries must be >= 1")
    # Match common MATLAB-style tie between odd kernel size and Gaussian sigma.
    sigma_y = 0.5 * float(ky - 1)
    sigma_x = 0.5 * float(kx - 1)
    arr = np.asarray(density_map)
    if arr.ndim == 1:
        return gaussian_filter(arr, sigma=sigma_x)
    if arr.ndim == 2:
        return gaussian_filter(arr, sigma=(sigma_y, sigma_x))
    raise ValueError(f"density_map must be 1D or 2D, got shape {arr.shape}")
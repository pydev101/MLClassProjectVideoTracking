# Author: Taylor Lindley
# Date: 4/4/2026
#
# Visualisation helpers for crowd density maps.
#
# All plotting functions accept the same optional keyword arguments (figsize,
# title, alpha, cmap, show, block) so they compose consistently.  The three
# main entry points are:
#
#   plot_csrnet_density_map   – overlay CSRNet's predicted density on the image
#   plot_ground_truth_density – overlay ground-truth density on the image
#   plot_density_map          – standalone heatmap (no base image)
#
# Each delegates to ``plot_density_overlay`` for the actual matplotlib work.

from __future__ import annotations
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


def plot_image_with_density_map(
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

    The density map may have a different resolution than the image; matplotlib's
    ``extent`` parameter stretches it to cover the same pixel range so the
    overlay lines up correctly.

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
    if density_map is None:
        raise ValueError("density_map must be a 2D array, got None")
    dm = np.asarray(density_map, dtype=np.float64)
    if dm.ndim != 2:
        raise ValueError(f"density_map must be 2D for imshow, got shape {dm.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    # extent=[left, right, bottom, top] stretches the heatmap to the same
    # pixel coordinates as the base image regardless of resolution mismatch.
    ax.imshow(
        dm,
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

def plot_density_map(
        density_map: np.ndarray,
        *,
        gaussian_kernel_size: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6),
        cmap=CM.jet,
        show: bool = True,
        block: bool = False,
    ) -> Figure:
    """
    Plot a standalone density heatmap (no base image underneath).

    By default, no Gaussian blur is applied. gaussian_blur = (x, y) defines the kernel size for the Gaussian blur.

    Args:
        density_map: 2D array ``[H, W]``.
        gaussian_kernel_size: Kernel size for pre-display Gaussian smoothing.
        title: Optional axes title.
        figsize: Matplotlib figure size.
        cmap: Colormap for the heatmap.
        show: If True, call ``plt.show()``.
        block: Passed to ``plt.show(block=...)`` (default non-blocking).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    # Apply Gaussian blur if requested
    if gaussian_kernel_size is not None:
        blurred = gaussian_blur(np.asarray(density_map, dtype=np.float64), kernel_size=gaussian_kernel_size)
    else:
        blurred = np.asarray(density_map, dtype=np.float64)

    # Plot the density map
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(blurred, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show(block=block)
    return fig

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
    """Small wrapper for plotting ground truth data ontop of an image"""
    if title is None:
        count = float(np.sum(ground_truth_density))
        title = f"Ground Truth Density Map\nCount (sum): {count:.2f}"
    return plot_image_with_density_map(
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
    Light Gaussian smoothing for visualisation (not training).

    Sigma is derived from the kernel size with the MATLAB convention:
    ``sigma = 0.5 * (k - 1)``, so a 5x5 kernel gives sigma=2.0.

    Args:
        density_map: 1D or 2D array to smooth.
        kernel_size: ``(height, width)`` of the smoothing kernel.
        
    Returns:
        np.ndarray: Blurred density map with the same shape and dtype as the input.
    """
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    if ky < 1 or kx < 1:
        raise ValueError("kernel_size entries must be >= 1")
    sigma_y = 0.5 * float(ky - 1)
    sigma_x = 0.5 * float(kx - 1)
    arr = np.asarray(density_map)
    if arr.ndim == 1:
        return gaussian_filter(arr, sigma=sigma_x)
    if arr.ndim == 2:
        return gaussian_filter(arr, sigma=(sigma_y, sigma_x))
    raise ValueError(f"density_map must be 1D or 2D, got shape {arr.shape}")

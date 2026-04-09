"""
Standalone test / exploration script for the Mall dataset.

The Mall dataset stores ground-truth head locations in a ``.mat`` file
(``mall_gt.mat``) with a nested cell-array structure:

    mat_data['frame'][0, i]       -> struct for frame i (0-indexed in Python)
    mat_data['frame'][0, i]['loc'][0, 0]  -> Nx2 array of (x, y) head coords

This script:
  1. Loads the ground-truth .mat file.
  2. Reads a single frame image (``seq_NNNNNN.jpg``).
  3. Stamps a unit impulse at each head location to build a raw heatmap.
  4. Displays the heatmap and the image with head markers overlaid.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the MATLAB ground-truth file; returns a dict whose 'frame' entry is a
# (1, N_frames) object array of structs, one per video frame.
mat_data = sio.loadmat('mall_gt.mat')

# Pick a frame index (1-based in the filename, 0-based in Python arrays).
img_index = 940
img_path = f'./frames/seq_{img_index:06d}.jpg'

img = mpimg.imread(img_path)

# Extract the (x, y) head coordinates for this frame from the nested struct.
frame_data = mat_data['frame'][0, img_index - 1]
points = np.array(frame_data['loc'][0, 0])

print(f"Coordinates Shape: {points.shape}")   # Expected: (N, 2)
print(f"Total People: {points.shape[0]}")

def create_heatmap(points, img_shape, sigma=8):
    """Build a sparse impulse map: a 1 at every head coordinate.

    Each (x, y) pair gets a unit value placed at pixel (row=y, col=x).
    Optionally a Gaussian blur could be applied afterwards (commented out)
    to convert the impulses into a smooth density surface.
    """
    heatmap = np.zeros(img_shape, dtype=np.float32)

    for x, y in points:
        if int(y) < img_shape[0] and int(x) < img_shape[1]:
            heatmap[int(y), int(x)] += 1

    # Uncomment to produce a smooth density map:
    # heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap

img_h, img_w = img.shape[0], img.shape[1]
my_heatmap = create_heatmap(points, (img_h, img_w))

# Plot 1: raw heatmap (jet colourmap highlights head locations).
plt.imshow(my_heatmap, cmap='jet')

# Plot 2: original image with red stars at each annotated head.
plt.figure()
plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1], c='red', marker='*', s=10)
plt.title(f'Frame {img_index}')
plt.show()

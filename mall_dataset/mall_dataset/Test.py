import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Load the .mat file
# Note: loadmat returns a dictionary
mat_data = sio.loadmat('mall_gt.mat')

# 2. Setup file path and index
# Python uses f-strings or % for formatting; 06d ensures 6-digit padding
img_index = 940
img_path = f'./frames/seq_{img_index:06d}.jpg'

# 3. Read the image
img = mpimg.imread(img_path)

# 4. Access the coordinates
# In MATLAB, 'frame' was a cell array. In Python, it's often a nested numpy array.
# You may need to check the exact structure: mat_data['frame'][0, img_index-1]
# Assuming XY is a (N, 2) array of coordinates:
frame_data = mat_data['frame'][0, img_index - 1]
points = np.array(frame_data['loc'][0, 0])

print(f"Coordinates Shape: {points.shape}") # Should be (N, 2)
print(f"Total People: {points.shape[0]}")

def create_heatmap(points, img_shape, sigma=8):
    # 1. Create an empty black image
    heatmap = np.zeros(img_shape, dtype=np.float32)

    # 2. Place a '1' at every head coordinate
    for x, y in points:
        if int(y) < img_shape[0] and int(x) < img_shape[1]:
            heatmap[int(y), int(x)] += 1

    # 3. Apply Gaussian blur to spread the "heat"
    # heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap

img_h, img_w = img.shape[0], img.shape[1]
my_heatmap = create_heatmap(points, (img_h, img_w))
plt.imshow(my_heatmap, cmap='jet')

# 5. Visualize
plt.figure()
plt.imshow(img)
plt.scatter(points[:, 0], points[:, 1], c='red', marker='*', s=10)
plt.title(f'Frame {img_index}')
plt.show()
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg

typeData = "test_data"
N = 4

mat = sio.loadmat(f'./{typeData}/ground_truth/GT_IMG_{N}.mat')
img = mpimg.imread(f'./{typeData}/images/IMG_{N}.jpg')

points = mat['image_info'][0, 0][0, 0][0]

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

img = img.copy()

fig, ax = plt.subplots()
ax.imshow(img)

for x, y in points:
    if int(y) < img.shape[0] and int(x) < img.shape[1]:
        # print(x, y)
        circle = plt.Circle((x, y), 1, color='r')
        ax.add_patch(circle)

plt.show()
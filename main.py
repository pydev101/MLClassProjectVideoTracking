from pathlib import Path
from scipy.ndimage import gaussian_filter

from csrnet import *
from plotting_tools import *


def example():
    weights_a = Path("./models/csr_net_base/PartBmodel_best.pth")
    img_path = Path("./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_2.jpg")
    # img_path = Path("./test_images/1.jpeg")

    if not img_path.exists():
        print(f"Warning: Could not find {img_path}. Make sure the image exists to run the example.")
        return

    # To use random (untrained) weights for a quick sanity check:
    # model = CSRNet(load_weights=True)
    # To use a trained checkpoint:
    model = load_csrnet_model(weights_a, map_location="cpu")
    img, density_map, estimated_crowd_count = load_and_process_image_with_csrnet(model, img_path)

    # Ground-truth .mat files live in a sibling "ground_truth" directory.
    gt_path = img_path.parent.parent / "ground_truth" / f"GT_{img_path.stem}.mat"

    if gt_path.exists():
        gt_density = load_ground_truth(gt_path, image_path=img_path)
        plot_ground_truth_density(img, gt_density)

    # The network output is 1/8 of the input size; up-scale for display.
    h, w = img.size[1], img.size[0]
    interp_density_map = bilinear_interpolation(density_map, (h, w))

    # Plot the image and density map 
    title = f"CSRNet Crowd Density Map\nEstimated Count: {estimated_crowd_count:.2f}"
    plot_image_with_density_map(img, interp_density_map, title=title, block=False)
    # plot just the density map
    plot_density_map(interp_density_map, block=True, gaussian_kernel_size=(3, 3))



if __name__ == "__main__":
    example()

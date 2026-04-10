from pathlib import Path
from scipy.ndimage import gaussian_filter

from csrnet import *
from plotting_tools import *


def example():
    # checkpoint = Path("./models/csr_net_base/PartBmodel_best.pth")
    # checkpoint = Path("./models/csr_net_base/PartAmodel_best.pth")
    # checkpoint = Path("./testmodel_best.pth")
    # checkpoint = Path("./all_model_best.pth")
    checkpoint = Path("testcheckpoint.pth")
    img_path_base = Path("./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_6.jpg")
    model_name = "PartA_downloaded"
    name = "IMG_6"
    save_images = False

    img_path = img_path_base.parent / f"{name}.jpg"
    name += f"_{model_name}"
    # img_path = Path("./ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/images/IMG_1.jpg")
    # img_path = Path("./test_images/1.jpeg"

    if not img_path.exists():
        print(f"Warning: Could not find {img_path}. Make sure the image exists to run the example.")
        return

    # To use random (untrained) weights for a quick sanity check:
    # model = CSRNet(load_weights=True)
    # To use a trained checkpoint:
    model = load_csrnet_model(checkpoint, map_location="cpu")
    img, density_map, estimated_crowd_count = load_and_process_image_with_csrnet(model, img_path)

    # Ground-truth .mat files live in a sibling "ground_truth" directory.
    gt_path = img_path.parent.parent / "ground_truth" / f"GT_{img_path.stem}.mat"

    if gt_path.exists():
        gt_density = tech_load_ground_truth_density_map(gt_path, image_path=img_path)
        plot_ground_truth_density(img, gt_density, block=False, save=save_images, name=name+"_gt")

    # The network output is 1/8 of the input size; up-scale for display.
    h, w = img.size[1], img.size[0]
    interp_density_map = bilinear_interpolation(density_map, (h, w))

    # Plot the image and density map 
    title = f"CSRNet Crowd Density Map\nEstimated Count: {estimated_crowd_count:.2f}"
    plot_image_with_density_map(img, interp_density_map, title=title, block=False, save=save_images, name=name)
    # plot just the density map
    plot_density_map(interp_density_map, block=True, gaussian_kernel_size=(2, 2), save=save_images, name=name)


def random_image_example():
    """
    This is an example of running inference on a random image and plotting the density map along with the image
    """
    # load a random image from the dataset
    checkpoint = Path("./all_model_best.pth")
    model = load_csrnet_model(checkpoint, map_location="cpu")
    img_path = Path("./test_images/crowd.jpg")
    name = "crowd"
    save_images = False

    img, density_map, estimated_crowd_count = load_and_process_image_with_csrnet(model, img_path)
    h, w = img.size[1], img.size[0]
    interp_density_map = bilinear_interpolation(density_map, (h, w))
    title = f"CSRNet Crowd Density Map\nEstimated Count: {estimated_crowd_count:.2f}"
    plot_image_with_density_map(img, interp_density_map, title=title, block=False, save=save_images, name=name)
    plot_density_map(interp_density_map, block=False, gaussian_kernel_size=(2, 2), save=save_images, name=name)
    gt_path = "./tools/labeling/labeled_data/ground_truth/json/crowd_20260410T060438Z.json"
    gt_density_map = labeltool_load_ground_truth_density_map(gt_path)
    name += "_gt"
    gt_title = f"Crowd Density Map (Ground Truth)\nCount: {np.sum(gt_density_map):.2f}"
    plot_image_with_density_map(img, gt_density_map, title=gt_title, block=False, save=save_images, name=name)
    plot_density_map(gt_density_map, title=gt_title, block=True, gaussian_kernel_size=(2, 2), save=save_images, name=name)

if __name__ == "__main__":
    # example()
    random_image_example()

from pathlib import Path

from csrnet import load_and_process_image_with_csrnet, load_csrnet_model, load_ground_truth
from plotting_tools import plot_csrnet_density_map, plot_ground_truth_density


def example():
    weights_a = Path("./models/csr_net_base/PartBmodel_best.pth")
    img_path = Path("./ShanghaiTech_Crowd_Counting_Dataset/part_A_final/test_data/images/IMG_2.jpg")

    if not img_path.exists():
        print(f"Warning: Could not find {img_path}. Make sure the image exists to run the example.")
        return

    model = load_csrnet_model(weights_a, map_location="cpu")
    img, density_map, estimated_crowd_count = load_and_process_image_with_csrnet(model, img_path)

    gt_path = img_path.parent.parent / "ground_truth" / f"GT_{img_path.stem}.mat"
    if gt_path.exists():
        gt_density = load_ground_truth(gt_path, image_path=img_path)
        plot_ground_truth_density(img, gt_density)

    plot_csrnet_density_map(img, density_map, estimated_crowd_count, block=True)



if __name__ == "__main__":
    example()

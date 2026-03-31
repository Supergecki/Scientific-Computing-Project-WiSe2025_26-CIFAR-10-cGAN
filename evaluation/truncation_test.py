import os
import sys
import yaml
import argparse
import torch
from torchvision.utils import save_image
from scipy.stats import truncnorm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.cgan import CGAN


def get_truncated_noise(batch_size, z_dim, truncation_threshold):
    """Generates truncated noise from a normal distribution."""
    truncated_z = truncnorm.rvs(
        -truncation_threshold, truncation_threshold, size=(batch_size, z_dim)
    )
    return torch.FloatTensor(truncated_z)


def run_truncation_ablation(
    model, latent_dim, num_classes, device, thresholds=[0.5, 1.0, 2.0]
):
    """
    Generates a comparison grid showing the effect of different truncation thresholds.
    Each row corresponds to a different threshold.
    """
    model.eval()
    output_dir = "./results/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        all_imgs = []
        # Generate one image per class for the grid width
        labels = torch.arange(num_classes, device=device)

        print("Generating ablation grid...")
        for threshold in thresholds:
            print(f"Sampling with psi = {threshold}")
            z = get_truncated_noise(num_classes, latent_dim, threshold).to(device)
            imgs = model(z, labels)
            all_imgs.append(imgs)

        # Stack rows vertically
        grid = torch.cat(all_imgs, dim=0)
        output_path = os.path.join(output_dir, "truncation_ablation.png")

        save_image(
            grid,
            output_path,
            nrow=num_classes,
            normalize=True,
            value_range=(-1, 1),
        )

        print("-" * 50)
        print(f"Success! Saved truncation ablation grid to: {output_path}")
        print(f"Row 1 (Heavy Truncation):  psi = {thresholds[0]}")
        print(f"Row 2 (Moderate Truncation): psi = {thresholds[1]}")
        print(f"Row 3 (Light/No Truncation): psi = {thresholds[2]}")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Truncation Trick Ablation Study")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/biggan_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="Path to checkpoint .pth file"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    print("Loading model weights...")
    cgan = CGAN(config)
    cgan.load(args.weights)
    cgan.generator.to(device)

    # Execute test
    run_truncation_ablation(
        cgan.generator,
        config["model"]["latent_dim"],
        config["data"]["num_classes"],
        device,
    )

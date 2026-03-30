import torch
import numpy as np
import argparse
import yaml
import os
import sys
from torchvision.utils import save_image

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.cgan import CGAN


def slerp(val, low, high):
    """Spherical interpolation for latent vectors"""
    omega = torch.acos(
        torch.clamp(
            torch.sum(
                low
                / torch.norm(low, dim=1, keepdim=True)
                * high
                / torch.norm(high, dim=1, keepdim=True),
                dim=1,
            ),
            -1,
            1,
        )
    )
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return (
        torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high
    )


def generate_interpolations(model, latent_dim, num_classes, device, steps=10):
    """Generates an interpolation grid: smooth transition between two noise vectors."""
    model.eval()
    os.makedirs("./results/visualizations", exist_ok=True)

    with torch.no_grad():
        # Pick 5 random classes
        classes = torch.randint(0, num_classes, (5,), device=device)

        all_imgs = []
        for c in classes:
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)

            # Interpolate
            alphas = torch.linspace(0, 1, steps, device=device)
            zs = torch.cat([slerp(a, z1, z2) for a in alphas])
            labels = torch.full((steps,), c.item(), dtype=torch.long, device=device)

            imgs = model(zs, labels)
            all_imgs.append(imgs)

        grid = torch.cat(all_imgs, dim=0)
        save_image(
            grid,
            "./results/visualizations/interpolation.png",
            nrow=steps,
            normalize=True,
            value_range=(-1, 1),
        )
        print("Saved interpolation grid to ./results/visualizations/interpolation.png")


def generate_class_variation(model, latent_dim, num_classes, device):
    """Generates a grid where each row is the same noise vector, but different classes."""
    model.eval()
    os.makedirs("./results/visualizations", exist_ok=True)

    with torch.no_grad():
        # 5 random latent vectors
        zs = torch.randn(5, latent_dim, device=device)

        all_imgs = []
        for z in zs:
            z_repeated = z.unsqueeze(0).repeat(num_classes, 1)
            labels = torch.arange(num_classes, device=device)

            imgs = model(z_repeated, labels)
            all_imgs.append(imgs)

        grid = torch.cat(all_imgs, dim=0)
        save_image(
            grid,
            "./results/visualizations/class_variation.png",
            nrow=num_classes,
            normalize=True,
            value_range=(-1, 1),
        )
        print(
            "Saved class variation grid to ./results/visualizations/class_variation.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./../config/biggan_conig.yaml"
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="Path to checkpoint .pth file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cgan = CGAN(config)
    cgan.load(args.weights)
    cgan.generator.to(device)

    generate_interpolations(
        cgan.generator,
        config["model"]["latent_dim"],
        config["data"]["num_classes"],
        device,
    )
    generate_class_variation(
        cgan.generator,
        config["model"]["latent_dim"],
        config["data"]["num_classes"],
        device,
    )

import torch
import numpy as np
import argparse
import yaml
import os
import sys
import matplotlib.pyplot as plt
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
        return (1.0 - val) * low + val * high
    return (
        torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high
    )


def plot_training_history(log_path, output_dir="./results/visualizations"):
    """Parses the text log file and plots Loss and FID histories."""
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}. Skipping plots.")
        return

    # Load data skipping the header: Epoch, D_Loss, G_Loss, D_Acc, FID
    try:
        data = np.genfromtxt(log_path, delimiter=",", skip_header=1)
        # If the file has only one line, we need to reshape
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    epochs = data[:, 0]
    d_loss = data[:, 1]
    g_loss = data[:, 2]
    fid = data[:, 4]

    os.makedirs(output_dir, exist_ok=True)

    # Plot Loss History
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, d_loss, label="Discriminator Loss", alpha=0.8)
    plt.plot(epochs, g_loss, label="Generator Loss", alpha=0.8)
    plt.title("GAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close()

    # Plot FID History (Filter out NaNs because FID is calculated every 5 epochs)
    mask = ~np.isnan(fid)
    if np.any(mask):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[mask], fid[mask], marker="o", color="orange", label="FID Score")
        plt.title("FID Score History (Lower is Better)")
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(output_dir, "fid_history.png"))
        plt.close()
        print(f"Plots saved to {output_dir}")


def generate_interpolations(model, latent_dim, num_classes, device, steps=10):
    model.eval()
    os.makedirs("./results/visualizations", exist_ok=True)
    with torch.no_grad():
        classes = torch.randint(0, num_classes, (5,), device=device)
        all_imgs = []
        for c in classes:
            z1 = torch.randn(1, latent_dim, device=device)
            z2 = torch.randn(1, latent_dim, device=device)
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
        print("Saved interpolation grid.")


def generate_class_variation(model, latent_dim, num_classes, device):
    model.eval()
    os.makedirs("./results/visualizations", exist_ok=True)
    with torch.no_grad():
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
        print("Saved class variation grid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="./../config/biggan_config.yaml"
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True, help="Path to checkpoint .pth file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = config["model"].get("architecture", "baseline")

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

    # We look for the file in ./results/ based on the architecture prefix
    log_file = f"./results/{prefix}_training_log.txt"
    plot_training_history(log_file)

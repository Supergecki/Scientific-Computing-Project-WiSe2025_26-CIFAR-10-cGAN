import copy
import sys
import argparse
import yaml
import os
import random
import numpy as np
from scipy.stats import truncnorm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import torch
from torchvision.utils import save_image
from data.dataloader import get_dataloaders
from models.cgan import CGAN
from evaluation.evaluate import calculate_fid
from training.losses import (
    get_baseline_loss,
    hinge_loss_discriminator,
    hinge_loss_generator,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config):
    set_seed(config["data"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("./results", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    train_loader, test_loader = get_dataloaders(config)

    latent_dim = config["model"]["latent_dim"]
    num_classes = config["data"]["num_classes"]
    lr_g = config["training"]["lr_g"]
    lr_d = config["training"]["lr_d"]
    beta1 = config["training"]["beta1"]
    beta2 = config["training"]["beta2"]
    num_epochs = config["training"]["num_epochs"]
    loss_type = config["training"].get("loss", "bce")
    use_ema = config["training"].get("use_ema", False)
    n_critic = config["training"].get("n_critic", 1)  # Get the training ratio

    cgan_model = CGAN(config)

    g_optimizer = torch.optim.Adam(
        cgan_model.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
    )
    d_optimizer = torch.optim.Adam(
        cgan_model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
    )
    cgan_model.generator_optimizer = g_optimizer
    cgan_model.discriminator_optimizer = d_optimizer

    cgan_model.generator.to(device)
    cgan_model.discriminator.to(device)

    if loss_type == "bce":
        criterion = get_baseline_loss().to(device)

    if use_ema:
        print("Using Exponential Moving Average (EMA)")
        ema_generator = copy.deepcopy(cgan_model.generator).to(device)
        ema_generator.eval()
        for param in ema_generator.parameters():
            param.requires_grad = False

    prefix = config["model"].get("architecture", "baseline")
    log_file_path = f"./results/{prefix}_training_log.txt"
    with open(log_file_path, "w") as f:
        f.write("Epoch, D_Loss, G_Loss, D_Acc, FID\n")

    fixed_z = torch.randn(25, latent_dim, device=device)
    fixed_labels = torch.randint(0, num_classes, (25,), device=device)

    print(f"Starting Training Loop with {n_critic}:1 D:G ratio...")

    for epoch in range(num_epochs):
        epoch_d_loss, epoch_g_loss, epoch_d_acc = 0.0, 0.0, 0.0
        g_updates = 0  # Track how many times G actually updated

        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            # =================================================
            # Train Discriminator

            cgan_model.discriminator_optimizer.zero_grad()

            real_logits = cgan_model.discriminate(real_imgs, real_labels)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = cgan_model.generate(z, real_labels)
            fake_logits = cgan_model.discriminate(fake_imgs.detach(), real_labels)

            if loss_type == "bce":
                valid = torch.ones(batch_size, 1, device=device)
                fake_target = torch.zeros(batch_size, 1, device=device)
                d_loss = (
                    criterion(real_logits, valid) + criterion(fake_logits, fake_target)
                ) / 2
            elif loss_type == "hinge":
                d_loss = hinge_loss_discriminator(real_logits, fake_logits)

            d_loss.backward()
            cgan_model.discriminator_optimizer.step()

            real_acc = (real_logits > 0).float().mean().item()
            fake_acc = (fake_logits < 0).float().mean().item()

            epoch_d_loss += d_loss.item()
            epoch_d_acc += (real_acc + fake_acc) / 2

            # =======================================================================
            # Train Generator
            if (i + 1) % n_critic == 0:
                cgan_model.generator_optimizer.zero_grad()

                z_gen = torch.randn(batch_size, latent_dim, device=device)
                labels_gen = torch.randint(0, num_classes, (batch_size,), device=device)

                new_fake_imgs = cgan_model.generate(z_gen, labels_gen)
                gen_logits = cgan_model.discriminate(new_fake_imgs, labels_gen)

                if loss_type == "bce":
                    valid_gen = torch.ones(batch_size, 1, device=device)
                    g_loss = criterion(gen_logits, valid_gen)
                elif loss_type == "hinge":
                    g_loss = hinge_loss_generator(gen_logits)

                g_loss.backward()
                cgan_model.generator_optimizer.step()

                epoch_g_loss += g_loss.item()
                g_updates += 1

                # ===============================================================================
                # 3. Update EMA Model

                if use_ema:
                    decay = config["training"].get("ema_decay", 0.999)
                    with torch.no_grad():
                        for ema_param, param in zip(
                            ema_generator.parameters(),
                            cgan_model.generator.parameters(),
                        ):
                            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
                        for ema_buf, buf in zip(
                            ema_generator.buffers(), cgan_model.generator.buffers()
                        ):
                            ema_buf.data.copy_(buf.data)

        # Epoch Logging
        avg_d_loss = epoch_d_loss / len(train_loader)
        # Avoid division by zero if dataset is very small
        avg_g_loss = epoch_g_loss / max(1, g_updates)
        avg_d_acc = epoch_d_acc / len(train_loader)

        current_fid = float("nan")
        eval_model = ema_generator if use_ema else cgan_model.generator

        if (epoch + 1) % 5 == 0:
            print(f"Calculating FID for Epoch {epoch + 1}...")
            current_fid = calculate_fid(
                eval_model, test_loader, latent_dim, device, num_images=5000
            )

        with open(log_file_path, "a") as f:
            f.write(
                f"{epoch + 1}, {avg_d_loss:.4f}, {avg_g_loss:.4f}, {avg_d_acc:.4f}, {current_fid:.2f}\n"
            )

        print(
            f"[Epoch {epoch+1}/{num_epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}] [D Acc: {avg_d_acc:.4f}] [FID: {current_fid:.2f}]"
        )

        cgan_model.epoch = epoch + 1
        cgan_model.loss = (avg_d_loss, avg_g_loss)

        if (epoch + 1) % 5 == 0:
            cgan_model.save(f"./checkpoints/{prefix}_epoch_{epoch+1}.pth")

        arch_type = config["model"].get("architecture", "baseline")
        # The evaluation/visualization block
        eval_model.eval()
        with torch.no_grad():
            if arch_type == "biggan":
                # Truncation Trick: Sample from truncated normal (between -1.5 and 1.5) (only for biggan architecture)
                # This improves quality at the cost of variety
                trunc_noise = truncnorm.rvs(-1.5, 1.5, size=(25, latent_dim))
                eval_z = torch.tensor(trunc_noise, dtype=torch.float32, device=device)
                print(f"Applying Truncation Trick for {arch_type} visualization.")
            else:
                # Use standard normal for Baseline or ResNet
                eval_z = torch.randn(25, latent_dim, device=device)

            eval_imgs = eval_model(eval_z, fixed_labels)

        eval_model.train()

        save_image(
            eval_imgs.data,
            f"./results/{prefix}_epoch_{epoch+1}.png",
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )

    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training loop of the CIFAR-10 cGAN"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/improved_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default=None,
        choices=["baseline", "sn_cnn", "resnet"],
        help="Override architecture",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if args.arch:
        config["model"]["architecture"] = args.arch

    train(config)

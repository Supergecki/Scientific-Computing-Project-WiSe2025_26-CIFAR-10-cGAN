import sys
from losses import get_baseline_loss, get_lsgan_loss
import argparse
import yaml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision.utils import save_image
from data.dataloader import get_dataloaders
from models.cgan import CGAN
from evaluation.evaluate import calculate_fid


def set_seed(seed=42):
    """
    Sets the seed for reproducibility as specified by the project description.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config):
    """
    Main Training Loop for CGAN.

    Args:
        config (dict): Dictionary containing all hyperparameters loaded from YAML configuration file.
    """

    set_seed(config["data"].get("seed", 42))  # Set random seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting setup on device: {device}")

    # Create directories for outputs
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # Initialize DataLoaders
    train_loader, test_loader = get_dataloaders(config)

    # Extract Model and Training Hyperparameters
    latent_dim = config["model"]["latent_dim"]
    embed_dim = config["model"]["embed_dim"]
    gen_channels = config["model"]["generator_channels"]
    disc_channels = config["model"]["discriminator_channels"]
    num_classes = config["data"]["num_classes"]
    image_size = config["data"]["image_size"]

    lr_g = config["training"]["lr_g"]
    lr_d = config["training"]["lr_d"]
    beta1 = config["training"]["beta1"]
    beta2 = config["training"]["beta2"]
    num_epochs = config["training"]["num_epochs"]

    # Initialize CGAN model
    # For now, setting optimizers to None. Since torch.optim requires model parameters at creation, they will be initialized once the model is instantiated. (this will be fixed/improved later, not very complicated)

    cgan_model = CGAN(
        generator_channel_config=gen_channels,
        discriminator_channel_config=disc_channels,
        generator_optimizer=None,
        discriminator_optimizer=None,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        num_classes=num_classes,
        image_size=image_size,
        is_baseline=config.get("is_baseline", False)
    )

    # Now that the model is created, we attach the optimizers
    g_optimizer = torch.optim.Adam(
        cgan_model.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
    )

    d_optimizer = torch.optim.Adam(
        cgan_model.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
    )

    cgan_model.generator_optimizer = g_optimizer
    cgan_model.discriminator_optimizer = d_optimizer

    # Move models to devide and define the loss
    cgan_model.generator.to(device)
    cgan_model.discriminator.to(device)
    criterion = get_baseline_loss().to(
        device
    )  # later use loss from /training/losses.py

    prefix = "baseline" if config.get("is_baseline", False) else "improved"
    log_file_path = f"./results/{prefix}_training_log.txt"
    with open(log_file_path, "w") as f:
        f.write("Epoch, D_Loss, G_Loss, D_Acc, FID \n")

    # create fixed noise and labels to see how exactly the same images evolve
    fixed_z = torch.randn(25, latent_dim, device=device)
    # generate 25 labels: 5 classes, 5 examples each
    fixed_labels = torch.randint(0, num_classes, (25,), device=device)

    print("Starting Training lOop...")

    # Training Loop
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_acc = 0.0

        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            valid = torch.ones(batch_size, 1, device=device)  # one for valid
            fake = torch.zeros(batch_size, 1, device=device)  # zero for fake


            # Train Discriminator
            cgan_model.discriminator_optimizer.zero_grad()
            real_logits = cgan_model.discriminate(real_imgs, real_labels)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = cgan_model.generate(z, real_labels)
            fake_logits = cgan_model.discriminate(fake_imgs.detach(), real_labels)


            if config.get("is_baseline", False):
                # basline: bce loss

                d_real_loss = criterion(real_logits, valid)
                d_fake_loss = criterion(fake_logits, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2
            else:
                from losses import hinge_loss_discriminator
                d_loss = hinge_loss_discriminator(real_logits, fake_logits)


            d_loss.backward()
            cgan_model.discriminator_optimizer.step()


            real_acc = (real_logits > 0).float().mean().item()
            fake_acc = (fake_logits < 0).float().mean().item()


            # Train Generator
            cgan_model.generator_optimizer.zero_grad()
            gen_logits = cgan_model.discriminate(fake_imgs, real_labels)

            if config.get("is_baseline", False):

                g_loss = criterion(gen_logits, valid)
            else:
                # improved hinge loss
                from losses import hinge_loss_generator
                g_loss = hinge_loss_generator(gen_logits)


            g_loss.backward()
            cgan_model.generator_optimizer.step()

            # Track losses for epoch averaging
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_acc += (real_acc + fake_acc) / 2

        # Print training progress once per epoch
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_acc = epoch_d_acc / len(train_loader)

        current_fid = float("nan")
        if (epoch + 1) % 5 == 0:
            print(f"Calculating FID for Epoch {epoch + 1}")
            current_fid = calculate_fid(
                cgan_model.generator, test_loader, latent_dim, device, num_images=5000
            )

        with open(log_file_path, "a") as f:
            f.write(
                f"{epoch + 1}, {avg_d_loss:.4f}, {avg_g_loss:.4f}, {avg_d_acc:.4f}, {current_fid:.2f}\n"
            )

        print(
            f"[Epoch {epoch+1}/{num_epochs}] [D loss: {avg_d_loss:.4f}] [G loss: {avg_g_loss:.4f}] [D Acc: {avg_d_acc:.4f}] [FID: {current_fid:.2f}]"
        )

        # Save metrics inside the model
        cgan_model.epoch = epoch + 1
        cgan_model.loss = (avg_d_loss, avg_g_loss)

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Differentiates checkpoint names based on the config
            # prefix = "baseline" if config.get("is_baseline", False) else "improved"
            cgan_model.save(f"./checkpoints/{prefix}_epoch_{epoch+1}.pth")

        cgan_model.generator.eval()  # set to eval mode for generation
        with torch.no_grad():
            eval_imgs = cgan_model.generate(fixed_z, fixed_labels)
        cgan_model.generator.train()  # return to train mode

        # Save generated images
        # value_range(-1, 1) denormalize images back to [0, 1] for proper saving
        save_image(
            eval_imgs.data,
            f"./results/epoch_{epoch+1}.png",
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )

    print("Training completed.")


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))
    os.chdir("..")

    # parsing command line arguments, currently used for:
    # - choosing the config file (defaults to `improved_config.yaml`, use '-b' flag to use `baseline_config.yaml`
    #                             or '-c' flag to specify another config file using its path)
    parser = argparse.ArgumentParser(
        description="Run training loop of the CIFAR-10 cGAN"
    )
    parser.add_argument(
        "-b",
        "--baseline",
        action="store_true",
        default=False,
        help="use baseline_config.yaml instead of improved_config.yaml",
    )

    parser.add_argument(
        "-c",
        "--config",
        action="store",
        default=None,
        help="use specified config file; overrides -b",
    )

    args = parser.parse_args()

    # Choose the correct config file
    config_file = "./config/improved_config.yaml"
    is_baseline = False
    if args.config is not None:
        config_file = args.config
    elif args.baseline:
        config_file = "./config/baseline_config.yaml"
        is_baseline = True

    print(f"Loading configuration from: {config_file}")

    # Load config file and save configs in 'config' dictionary
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Inject a flag so the script knows how to name the checkpoint files
    config["is_baseline"] = is_baseline

    # Run the training loop using the loaded dictionary
    train(config)
# Now configs are accessible via the dictionary, e.g. config['model']['embed_dim'] to get embedding dimension

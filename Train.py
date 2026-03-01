import torch
import torch.nn as nn
import os
from torchvision.utils import save_image


def train(dataloader, cgan_model, num_epochs=100, device='cuda'):
    """ Training Loop for cGAN.
    Args:
        dataloader: torch.utils.data.DataLoader, the dataloader for training data.
        cgan_model: cGAN model, the model to be trained.
        num_epochs: int, number of epochs to train.
        device: str, device to train on ('cuda' or 'cpu').
    """
    # create directory to save generated images
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    # vanilla GAN loss function: Binary Cross Entropy with Logits Loss
    criterion = nn.BCEwithLogitsLoss().to(device)
    cgan_model.generator.to(device)
    cgan_model.discriminator.to(device)

    print("Starting Training Loop...")

    # Training Loop
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            cgan_model.discriminator.optimizer.zero_grad()

            # Real images
            real_logits = cgan_model.discriminator(real_imgs, real_labels)
            d_real_loss = criterion(real_logits, valid)

            # Fake images
            z = torch.randn(batch_size, cgan_model.latent_dim, device=device)
            fake_imgs = cgan_model.generator(z, real_labels)
            fake_logits = cgan_model.discriminator(fake_imgs.detach(), real_labels)
            d_fake_loss = criterion(fake_logits, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            cgan_model.discriminator.optimizer.step()

            # Train Generator
            cgan_model.generator.optimizer.zero_grad()

            # Evaluate generator loss
            gen_logits = cgan_model.discriminator(fake_imgs, real_labels)
            g_loss = criterion(gen_logits, valid)
            g_loss.backward()
            cgan_model.generator.optimizer.step()

            # Print training progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save metrics and generated images
        cgan_model.epoch = epoch + 1
        cgan_model.loss = (d_loss.item(), g_loss.item())
        save_image(fake_imgs.data[:25], f"./results/{epoch}.png", nrow=5, normalize=True, range=(-1, 1))

        # Save model checkpoint every 5 epochs
        if epoch % 5 == 0:
            cgan_model.save_checkpoint(f"./checkpoints/baseline_epoch_{epoch}.pth")

    print("Training completed.")

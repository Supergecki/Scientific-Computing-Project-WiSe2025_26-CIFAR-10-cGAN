import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from visualize import plot_losses

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

    # Lists to store losses for plotting
    d_losses_history = []
    g_losses_history = []

    # vanilla GAN loss function: Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss().to(device)
    cgan_model.generator.to(device)
    cgan_model.discriminator.to(device)

    print("Starting Training Loop...")
    
    # Training Loop
    for epoch in range(num_epochs):
        # Initialize epoch losses
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            real_labels = labels.to(device)

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            cgan_model.discriminator_optimizer.zero_grad()

            # Real images
            real_logits = cgan_model.discriminator(real_imgs, real_labels)
            d_real_loss = criterion(real_logits, valid)

            # Fake images
            z = torch.randn(batch_size, cgan_model.generator.nz, device=device)
            fake_imgs = cgan_model.generator(z, real_labels)
            fake_logits = cgan_model.discriminator(fake_imgs.detach(), real_labels)
            d_fake_loss = criterion(fake_logits, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            epoch_d_loss += d_loss.item()
            cgan_model.discriminator_optimizer.step()

            # Train Generator
            cgan_model.generator_optimizer.zero_grad()
            
            # Evaluate generator loss
            gen_logits = cgan_model.discriminator(fake_imgs, real_labels)
            g_loss = criterion(gen_logits, valid)
            g_loss.backward()
            epoch_g_loss += g_loss.item()
            cgan_model.generator_optimizer.step()

            # Print training progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                
        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        d_losses_history.append(avg_d_loss)
        g_losses_history.append(avg_g_loss)
                
        # Save metrics and generated images
        cgan_model.epoch = epoch + 1
        cgan_model.loss = (d_loss.item(), g_loss.item())
        save_image(fake_imgs.data[:25], f"./results/{epoch}.png", nrow=5, normalize=True, value_range=(-1, 1))

        # Save model checkpoint every 5 epochs
        if epoch % 5 == 0:
            cgan_model.save(f"./checkpoints/baseline_epoch_{epoch}.pth")

    # Plot losses after training
    plot_losses(d_losses_history, g_losses_history, save_path='./results/loss_plot.png')
    print("Loss plot saved to './results/loss_plot.png'")

    print("Training completed.")

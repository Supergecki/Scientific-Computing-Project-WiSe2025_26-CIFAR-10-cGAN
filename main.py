import torch
import torch.optim as optim
import yaml
import random
import numpy as np

# importing custom modules
from data_loader import get_dataloaders
from Generator import Generator
from Discriminator import Discriminator
from CGAN import CGAN
from train import train

def set_seed(seed):
    """ Set random seed for reproducibility across various libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ''' Main function to set up and train the cGAN model. '''
    # 1. load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. set random seed and device
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. prepare data
    train_loader, _ = get_dataloaders(config)

    # 4. initialize cGAN model
    cgan = CGAN(
        generator_channel_config=config['model']['gen_channels'],
        discriminator_channel_config=config['model']['disc_channels'],
        generator_optimizer=None,      # will be defined after model initialization to ensure optimizers have access to model parameters
        discriminator_optimizer=None,  # same as above
        latent_dim=config['model']['latent_dim'],
        num_classes=10,
        image_size=config['data']['image_size']
    )

    # 5. define optimizers for generator and discriminator
    gen_opt = optim.Adam(
        cgan.generator.parameters(), 
        lr=config['training']['lr'], 
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    disc_opt = optim.Adam(
        cgan.discriminator.parameters(), 
        lr=config['training']['lr'], 
        betas=(config['training']['beta1'], config['training']['beta2'])
    )

    # save optimizers in the cGAN model so they can be accessed during training
    cgan.generator_optimizer = gen_opt
    cgan.discriminator_optimizer = disc_opt

    # 6. start training
    train(
        dataloader=train_loader,
        cgan_model=cgan,
        num_epochs=config['training']['num_epochs'],
        device=device
    )

if __name__ == "__main__":
    main()
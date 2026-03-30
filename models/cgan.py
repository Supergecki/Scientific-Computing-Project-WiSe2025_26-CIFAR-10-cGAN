import torch

from models import discriminator
from models.discriminator import Discriminator
from models.generator import Generator

from models.sn_cnn_discriminator import ImprovedDiscriminator as SNDiscriminator
from models.sn_cnn_generatror import ImprovedGenerator as SNGeneartor
from models.resnet_discriminator import ResNetDiscriminator
from models.resnet_generator import ResNetGenerator

from models.biggan_generator import BigGANGenerator
from models.biggan_discriminator import BigGANDiscriminator


class CGAN:
    """
    Unified cGAN model consisting of a generator and discriminator.

    :param generator_channel_config:
        List[int], number of channels per convolutional layer in the generator.
    :param discriminator_channel_config:
        List[int], number of channels per convolutional layer in the discriminator.
    :param generator_optimizer:
        torch Optimizer to use for the Generator
    :param discriminator_optimizer:
        torch Optimizer to use for the Discriminator
    :param latent_dim:
        int, dimension of the generator's latent space.
    :param embed_dim:
        int, dimension of the class label embedding (default: 50)
    :param num_classes:
        int, number of classes in the dataset (default: 10 for CIFAR-10)
    :param image_size:
        int, image size in pixels (image_size x image_size) (default: 32 for CIFAR-10)
    :param is_baseline:
        bool, that states whether baseline or improved architecture should be run
    """

    def __init__(self, config):
        self.config = config
        arch_type = config["model"].get("architecture", "baseline")

        gen_channels = config["model"]["generator_channels"]
        disc_channels = config["model"]["discriminator_channels"]
        latent_dim = config["model"]["latent_dim"]
        embed_dim = config["model"]["embed_dim"]
        num_classes = config["data"]["num_classes"]
        image_size = config["data"]["image_size"]

        if arch_type == "resnet":
            self.generator = ResNetGenerator(
                gen_channels, num_classes, image_size, latent_dim, embed_dim
            )
            self.discriminator = ResNetDiscriminator(
                disc_channels, num_classes, image_size, embed_dim
            )
        elif arch_type == "biggan":  # The new Final Boss
            self.generator = BigGANGenerator(
                gen_channels, num_classes, image_size, latent_dim, embed_dim
            )
            self.discriminator = BigGANDiscriminator(
                disc_channels, num_classes, image_size, embed_dim
            )
        elif arch_type == "baseline":
            self.generator = Generator(
                gen_channels, num_classes, image_size, latent_dim, embed_dim
            )
            self.discriminator = Discriminator(
                disc_channels, num_classes, image_size, embed_dim
            )
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

        self.epoch = 0
        self.loss = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def generate(self, z, labels):
        return self.generator(z, labels)

    def discriminate(self, x, labels):
        return self.discriminator(x, labels)

    def save(self, path):
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                "epoch": self.epoch,
                "loss": self.loss,
            },
            path,
        )

    def load(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, weights_only=True, map_location=device)

        # Load model weights
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        # Only load optimizer states if the optimizers have been initialized
        if self.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(
                checkpoint["generator_optimizer_state_dict"]
            )

        if self.discriminator_optimizer is not None:
            self.discriminator_optimizer.load_state_dict(
                checkpoint["discriminator_optimizer_state_dict"]
            )

        self.epoch = checkpoint.get("epoch", 0)
        self.loss = checkpoint.get("loss", None)

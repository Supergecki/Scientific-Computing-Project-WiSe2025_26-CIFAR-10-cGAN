import torch

# baseline imports
from models.discriminator import Discriminator
from models.generator import Generator

# first improved versions (spec norm) imports
from models.sn_cnn_discriminator import ImprovedDiscriminator as SNDiscriminator
from models.sn_cnn_generatror import ImprovedGenerator as SNGeneartor

# second improved versions (ResNet)
# to be included


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

        # extract architecture type
        arch_type = config["model"].get("architecture", "baseline")

        gen_channels = config["model"]["generator_channels"]
        disc_channels = config["model"]["discriminator_channels"]
        latent_dim = config["model"]["latent_dim"]
        embed_dim = config["model"]["embed_dim"]
        num_classes = config["data"]["num_classes"]
        image_size = config["data"]["image_size"]

        print(f"Using Architecture: {arch_type.upper()}")

        if arch_type == "baseline":
            self.generator = Generator(
                gen_channels, num_classes, image_size, latent_dim, embed_dim
            )
            self.discriminator = Discriminator(
                disc_channels, num_classes, image_size, embed_dim
            )
        elif arch_type == "sn_cnn":
            self.generator = SNGeneartor(
                gen_channels, num_classes, image_size, latent_dim, embed_dim
            )
            self.discriminator = SNDiscriminator(
                disc_channels, num_classes, image_size, embed_dim
            )
        elif arch_type == "resnet":
            # TODO
            raise NotImplementedError("ResNet not yet implemented")
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

        # save metrics in the class itself
        self.epoch = 0
        self.loss = None

        # save optimizers
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def generate(self, z, labels):
        """
        Calls the cGAN's generator's forward function.

        :param z:
            torch.Tensor, latent vector to generate from
        :param labels:
            torch.Tensor, class labels corresponding to the generation
        :return:
            generated image tensor
        """
        return self.generator(z, labels)

    def discriminate(self, x, labels):
        """
        Calls the cGAN's discriminator's forward function.

        :param x:
            torch.Tensor, input images in shape (3, image_size, image_size)
        :param labels:
            torch.Tensor, class labels in shape (batch_size,)
        :return:
            class logit (x was a real image or not)
        """
        return self.discriminator(x, labels)

    def save(self, path):
        """
        Saves the whole model to a given path.
        :param path:
            file path to save the model to
        :return:
            None
        """
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
        """
        Loads the whole model from a given checkpoint at a path.
        :param path:
            file path to load the model from
        :return:
            None (all attributes of the class will be changed in place)
        """
        checkpoint = torch.load(path, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.generator_optimizer.load_state_dict(
            checkpoint["generator_optimizer_state_dict"]
        )
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state_dict"]
        )
        self.epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]

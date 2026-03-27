import torch
from models.improved_discriminator import ImprovedDiscriminator
from models.improved_generator import ImprovedGenerator
from models.discriminator import Discriminator
from models.generator import Generator


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

    def __init__(
        self,
        generator_channel_config,
        discriminator_channel_config,
        generator_optimizer,
        discriminator_optimizer,
        latent_dim,
        embed_dim=50,
        num_classes=10,
        image_size=32,
        is_baseline=True,
    ):
        if is_baseline:
            self.generator = Generator(
                channel_config=generator_channel_config,
                num_classes=num_classes,
                image_size=image_size,
                latent_dim=latent_dim,
                embed_dim=embed_dim
            )

            self.discriminator = Discriminator(
                channel_config=discriminator_channel_config,
                num_classes=num_classes,
                image_size=image_size,
                embed_dim=embed_dim
            )
        else:
            self.generator = ImprovedGenerator(
                channel_config=generator_channel_config,
                num_classes=num_classes,
                image_size=image_size,
                latent_dim=latent_dim,
                embed_dim=embed_dim
            )

            self.discriminator = ImprovedDiscriminator(
                channel_config=discriminator_channel_config,
                num_classes=num_classes,
                image_size=image_size,
                embed_dim=embed_dim
            )

        # save metrics in the class itself
        self.epoch = 0
        self.loss = None

        # save optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

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

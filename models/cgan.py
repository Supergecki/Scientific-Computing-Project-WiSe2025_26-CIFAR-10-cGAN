from discriminator import Discriminator
from generator import Generator


class CGAN:
    """
    Unified cGAN model consisting of a generator and discriminator.

    :param generator_channel_config:
        List[int], number of channels per convolutional layer in the generator.
    :param discriminator_channel_config:
        List[int], number of channels per convolutional layer in the discriminator.
    :param latent_dim:
        int, dimension of the generator's latent space.
    :param num_classes:
        int, number of classes in the dataset (default: 10 for CIFAR-10)
    :param image_size:
        int, image size in pixels (image_size x image_size) (default: 32 for CIFAR-10)
    """
    def __init__(self, generator_channel_config, discriminator_channel_config,
                 latent_dim, num_classes=10, image_size=32):
        self.generator = Generator(generator_channel_config, latent_dim, num_classes, image_size)
        self.discriminator = Discriminator(discriminator_channel_config, num_classes, image_size)

    def generate(self, z):
        """
        Calls the cGAN's generator's forward function.

        :param z:
            latent vector to generate from
        :return:
            generated image
        """
        return self.generator.forward(z)

    def discriminate(self, x):
        """
        Calls the cGAN's discriminator's forward function.

        :param x:
            torch.Tensor, input images in shape (3, image_size, image_size)
        :return:
            scalar probability that x was a real image
        """
        return self.discriminator.forward(x)

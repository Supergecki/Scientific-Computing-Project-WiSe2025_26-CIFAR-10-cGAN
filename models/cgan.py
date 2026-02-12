import torch
from discriminator import Discriminator
from generator import Generator


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
    :param num_classes:
        int, number of classes in the dataset (default: 10 for CIFAR-10)
    :param image_size:
        int, image size in pixels (image_size x image_size) (default: 32 for CIFAR-10)
    """
    def __init__(self, generator_channel_config, discriminator_channel_config, generator_optimizer,
                 discriminator_optimizer, latent_dim, num_classes=10, image_size=32):
        self.generator = Generator(generator_channel_config, latent_dim, num_classes, image_size)
        self.discriminator = Discriminator(discriminator_channel_config, num_classes, image_size)

        # save metrics in the class itself
        self.epoch = 0
        self.loss = None

        # save optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

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

    def save(self, path):
        """
        Saves the whole model to a given path.
        :param path:
            file path to save the model to
        :return:
            None
        """
        torch.save({'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                    'epoch': self.epoch, 'loss': self.loss}, path)

    def load(self, path):
        """
        Loads the whole model from a given checkpoint at a path.
        :param path:
            file path to load the model from
        :return:
            None (all attributes of the class will be changed in place)
        """
        checkpoint = torch.load(path, weights_only=True)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

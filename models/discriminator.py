import torch.nn as nn


class Discriminator(nn.Module):
    """
    Convolutional neural network to implement the Discriminator part of the cGAN.

    __init__ parameters:
    :param channel_config:
        List[int], number of channels per convolutional layer.
    :param num_classes:
        int, number of classes in the dataset (default: 10 for CIFAR-10)
    :param image_size:
        int, image size in pixels (image_size x image_size) (default: 32 for CIFAR-10)
    """

    def __init__(self, channel_config, num_classes=10, image_size=32):
        super().__init__()  # call to __init__ of superclass nn.Module

        current_image_size = image_size  # keep track of how much the image size has been reduced by convolutions
        # Create convolutional layers
        layers = nn.Sequential()
        for i in range(len(channel_config)):  # iterate through channel config to generate layers one after another
            if i == 0:
                # First layer: Use convolution from 3 (rgb) layers to first channel number. padding=1 ensures we halve
                # image size exactly.
                layers.append(nn.Conv2d(3, channel_config[0], kernel_size=4, stride=2, padding=1))
            else:
                # Any other layer: Use convolution from preceding channel number to the one we have now.
                layers.append(nn.Conv2d(channel_config[i-1], channel_config[i],
                                                  kernel_size=4, stride=2, padding=1))
                # Also apply BatchNormalization since this is not the first layer
                layers.append(nn.BatchNorm2d(channel_config[i]))
            layers.append(nn.LeakyReLU(0.2))  # always apply LeakyReLU at the end
            current_image_size = int(current_image_size / 2)  # image size has been halved (rounded down)

        self.conv_layers = layers

        # Now flatten and project to single output
        self.flatten_layers = nn.Sequential(nn.Flatten(),
                                            nn.Linear(channel_config[-1] * current_image_size * current_image_size, 1))

    def forward(self, x):
        """
        Forward pass through the network.

        :param x:
            torch.Tensor, input images in shape (3, image_size, image_size)
        :return:
            scalar probability that x was a real image
        """
        y = self.conv_layers(x)
        y = self.flatten_layers(y)
        return y

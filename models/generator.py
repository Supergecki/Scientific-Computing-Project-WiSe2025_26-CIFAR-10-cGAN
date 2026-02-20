import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Convolutional neural network to implement the Generator part of the cGAN.

    __init__ parameters:
    :param latent_dim:
        int, dimension of the noise vector z (default: 100)
    :param embed_dim:
        int, dimension of the class label embedding (default: 50)
    :param channel_config:
        List[int], number of channels for the upsampling layers (e.g., [256, 128, 64])
    :param num_classes:
        int, number of classes in the dataset (default: 10)
    :param image_size:
        int, image size in pixels (image_size x image_size) (default: 32 for CIFAR-10)
    """

    def __init__(
        self,
        channel_config,
        num_classes=10,
        image_size=32,
        latent_dim=100,
        embed_dim=50,
    ):
        super().__init__()  # call to __init__ of superclass nn.Module

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Class label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Initial projection from combined (z + emb) to the 512 * 4 * 4 feature map
        self.project = nn.Linear(latent_dim + embed_dim, 512 * 4 * 4)

        # Upsampling layers
        layers = nn.Sequential()

        # Starting with 512 channels from the reshaped projection
        in_channels = 512

        for out_channels in channel_config:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(
                nn.ReLU(inplace=False)
            )  # inplace False for now, since it might corrupt computations using an improved architecture (e.g. WGAN-GP)
            in_channels = out_channels

        self.upsample_layers = layers

        # Maps the last channel count (64) to 3 RGB channels
        # To keep resolution at 32x32, we use a 3x3 convolution with padding=1
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, z, labels):
        """
        Forward pass through the network.

        :param z:
            torch.Tensor, latent noise vector of shape (batch_size, latent_dim)
        :param labels:
            torch.Tensor, class labels of shape (batch_size)
        :return:
            Generated image tensor of shape (batch_size, 3, 32, 32)
        """
        # Embed the labels
        emb = self.label_embedding(labels)

        # Concatenate noise and label embedding
        x = torch.cat(
            [z, emb], dim=1
        )  # Shape now: (batch_size, latent_dim + embed_dim)

        # Project and reshape to (batch_size, 512, 4, 4)
        x = self.project(x)
        x = x.view(-1, 512, 4, 4)

        # Apply upsampling layers
        x = self.upsample_layers(x)

        # Apply output layer to get 3 RGB channels
        x = self.output_layer(x)

        return x

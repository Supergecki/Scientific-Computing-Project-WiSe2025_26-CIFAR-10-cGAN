import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class ImprovedDiscriminator(nn.Module):
    def __init__(self, channel_config, num_classes=10, image_size=32, embed_dim=50):
        super().__init__()

        # embedding also get spectral normalized
        self.embedding = spectral_norm(nn.Embedding(num_classes, embed_dim))

        layers = []
        in_channels = 3
        current_image_size = image_size

        for out_channels in channel_config:

            # Apply spectral norm to convolutional layers
            layers.append(
                spectral_norm(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=4, stride=2, padding=1
                    )
                )
            )
            layers.append(nn.LeakyReLU(0.2))
            in_channels = out_channels
            current_image_size = current_image_size // 2

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        flattened_size = channel_config[-1] * current_image_size * current_image_size

        # mapping flattened spatial features to match the embed_dim
        self.feature_mapping = spectral_norm(nn.Linear(flattened_size, embed_dim))

        # final unconditional output layer
        self.output_layer = spectral_norm(nn.Linear(embed_dim, 1))

    def forward(self, x, labels):
        # get image features
        h = self.conv_layers(x)
        h = self.flatten(h)

        # map features to the same dimension as out label embeddings
        features = self.feature_mapping(h)  # shape: (batch_size, embed_dim)

        # compute unconditional logit
        uncond_logits = self.output_layer(features)  # shape: (batch_size, 1)

        # compute conditional logit using dot product
        emb = self.embedding(labels)  # shape: (batch_size, embed_dim)
        cond_logits = torch.sum(
            features * emb, dim=1, keepdim=True
        )  # shape: (batch_size, 1)

        # final score consists of the sum of both logits
        return uncond_logits + cond_logits

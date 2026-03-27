import torch
import torch.nn as nn


class ImprovedGenerator(nn.Module):
    def __init__(
        self,
        channel_config,
        num_classes=10,
        image_size=32,
        latent_dim=100,
        embed_dim=50,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.project = nn.Linear(latent_dim + embed_dim, channel_config[0] * 4 * 4)

        layers = []
        in_channels = channel_config[0]

        # iterate thourgh the remaining channels

        for out_channels in channel_config[1:]:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))

            # leaky ReLU instead of ReLU for better gradient throughput
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            in_channels = out_channels


        self.upsample_layers = nn.Sequential(*layers)

        self.output_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        emb = self.label_embedding(labels)
        x = torch.cat([z, emb], dim=1)

        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)

        x = self.upsample_layers(x)
        x = self.output_layers(x)
        return x

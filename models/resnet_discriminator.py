import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F


class OptimizedDiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.shortcut = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 1, padding=0)
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.activation(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)

        skip = F.avg_pool2d(x, 2)
        skip = self.shortcut(skip)
        return h + skip


class DiscResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.activation = nn.ReLU(inplace=False)

        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        if in_channels != out_channels or downsample:
            self.shortcut = spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, padding=0)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        skip = self.shortcut(x)
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)
        return h + skip


class ResNetDiscriminator(nn.Module):
    def __init__(self, channel_config, num_classes=10, image_size=32, embed_dim=128):
        super().__init__()

        final_channels = channel_config[-1]
        self.embedding = spectral_norm(nn.Embedding(num_classes, final_channels))

        layers = []
        layers.append(OptimizedDiscResBlock(3, channel_config[0]))
        in_channels = channel_config[0]

        for i, out_channels in enumerate(channel_config[1:]):
            downsample = i < 2
            layers.append(
                DiscResBlock(in_channels, out_channels, downsample=downsample)
            )
            in_channels = out_channels

        self.res_blocks = nn.Sequential(*layers)
        self.activation = nn.ReLU(inplace=False)
        self.output_layer = spectral_norm(nn.Linear(final_channels, 1))

    def forward(self, x, labels):
        h = self.res_blocks(x)

        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        uncond_logits = self.output_layer(h)

        emb = self.embedding(labels)
        cond_logits = torch.sum(h * emb, dim=1, keepdim=True)

        return uncond_logits + cond_logits

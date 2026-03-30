import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels // 2, 1))
        self.out = spectral_norm(nn.Conv2d(in_channels // 2, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        N = W * H

        proj_query = self.query(x).view(B, -1, N).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, N)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value(x).view(B, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, -1, W, H)
        out = self.out(out)

        return self.gamma * out + x


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
        skip = self.shortcut(F.avg_pool2d(x, 2))
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


class BigGANDiscriminator(nn.Module):
    def __init__(self, channel_config, num_classes=10, image_size=32, embed_dim=128):
        super().__init__()
        final_channels = channel_config[-1]
        self.embedding = spectral_norm(nn.Embedding(num_classes, final_channels))

        # 32x32 -> 16x16
        self.block1 = OptimizedDiscResBlock(3, channel_config[0])
        # Self-Attention at 16x16
        self.attention = SelfAttention(channel_config[0])
        # 16x16 -> 8x8
        self.block2 = DiscResBlock(
            channel_config[0], channel_config[1], downsample=True
        )
        # 8x8 -> 4x4
        self.block3 = DiscResBlock(
            channel_config[1], channel_config[2], downsample=True
        )
        # 4x4 -> 4x4
        self.block4 = DiscResBlock(
            channel_config[2], channel_config[3], downsample=False
        )

        self.activation = nn.ReLU(inplace=False)
        self.output_layer = spectral_norm(nn.Linear(final_channels, 1))
        self.apply(init_weights)

    def forward(self, x, labels):
        h = self.block1(x)
        h = self.attention(h)  # Apply attention!
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        uncond_logits = self.output_layer(h)
        emb = self.embedding(labels)
        cond_logits = torch.sum(h * emb, dim=1, keepdim=True)

        return uncond_logits + cond_logits

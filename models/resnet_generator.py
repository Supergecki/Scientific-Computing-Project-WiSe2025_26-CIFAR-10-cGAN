import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def init_weights(m):
    """Safe Orthogonal Initialization for GANs"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.gamma_proj = spectral_norm(nn.Linear(cond_dim, num_features))
        self.beta_proj = spectral_norm(nn.Linear(cond_dim, num_features))

        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x, cond_vector):
        out = self.bn(x)
        gamma = self.gamma_proj(cond_vector).view(-1, self.num_features, 1, 1)
        beta = self.beta_proj(cond_vector).view(-1, self.num_features, 1, 1)
        return out * gamma + beta


class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channels, cond_dim)
        self.activation = nn.ReLU(inplace=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.cbn2 = ConditionalBatchNorm2d(out_channels, cond_dim)
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                spectral_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
                ),
            )
        else:
            self.shortcut = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, cond_vector):
        h = self.cbn1(x, cond_vector)
        h = self.activation(h)
        h = self.upsample(h)
        h = self.conv1(h)

        h = self.cbn2(h, cond_vector)
        h = self.activation(h)
        h = self.conv2(h)

        skip = self.shortcut(x)
        return h + skip


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        channel_config,
        num_classes=10,
        image_size=32,
        latent_dim=128,
        embed_dim=128,
    ):
        super().__init__()

        self.shared_embedding = nn.Embedding(num_classes, embed_dim)
        self.base_channels = channel_config[0]
        self.project = spectral_norm(nn.Linear(latent_dim, self.base_channels * 4 * 4))

        self.res_blocks = nn.ModuleList()
        in_channels = self.base_channels
        cond_dim = embed_dim + latent_dim  # BigGAN Skip-z dimension

        for out_channels in channel_config:
            self.res_blocks.append(GenResBlock(in_channels, out_channels, cond_dim))
            in_channels = out_channels

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z, labels):
        emb = self.shared_embedding(labels)
        cond_vector = torch.cat([z, emb], dim=1)  # BIGGAN SKIP-Z TRICK

        x = self.project(z)
        x = x.view(x.size(0), self.base_channels, 4, 4)

        for block in self.res_blocks:
            x = block(x, cond_vector)

        x = self.output_layer(x)
        return x

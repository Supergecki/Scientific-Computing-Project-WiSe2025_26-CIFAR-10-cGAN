import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class SelfAttention(nn.Module):
    """Self-Attention Block (Non-Local Block)"""

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
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))

        self.cbn2 = ConditionalBatchNorm2d(out_channels, cond_dim)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, padding=0)),
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

        return h + self.shortcut(x)


class BigGANGenerator(nn.Module):
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

        cond_dim = embed_dim + latent_dim

        # 4x4 -> 8x8
        self.block1 = GenResBlock(channel_config[0], channel_config[1], cond_dim)
        # 8x8 -> 16x16
        self.block2 = GenResBlock(channel_config[1], channel_config[2], cond_dim)
        # Self-Attention at 16x16
        self.attention = SelfAttention(channel_config[2])
        # 16x16 -> 32x32
        self.block3 = GenResBlock(channel_config[2], channel_config[2], cond_dim)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(channel_config[2]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_config[2], 3, 3, padding=1),
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, z, labels):
        emb = self.shared_embedding(labels)
        cond_vector = torch.cat([z, emb], dim=1)

        x = self.project(z).view(z.size(0), self.base_channels, 4, 4)
        x = self.block1(x, cond_vector)
        x = self.block2(x, cond_vector)
        x = self.attention(x)  # Apply attention!
        x = self.block3(x, cond_vector)

        return self.output_layer(x)

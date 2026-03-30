import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-Attention block (Non-local block) from SA-GAN/BigGAN.
    Allows the network to look at all spatial locations to generate a single pixel,
    ensuring global consistency (e.g., symmetry).
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.key_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.value_conv = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        )
        self.out_conv = spectral_norm(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )

        # Learnable scale parameter, initialized to 0 so the block acts as an identity at first
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        N = width * height

        # Queries, Keys, Values
        proj_query = (
            self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)
        )  # B x N x C//8
        proj_key = self.key_conv(x).view(batch_size, -1, N)  # B x C//8 x N

        # Attention map
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(energy)  # B x N x N

        proj_value = self.value_conv(x).view(batch_size, -1, N)  # B x C//2 x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C//2 x N

        out = out.view(batch_size, -1, width, height)
        out = self.out_conv(out)

        return self.gamma * out + x


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization.
    Instead of learning static gamma and beta, it dynamically generates them from the class label.
    """

    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.gamma_proj = spectral_norm(nn.Linear(embed_dim, num_features))
        self.beta_proj = spectral_norm(nn.Linear(embed_dim, num_features))

        # Initialize to act as an identity mapping at the start
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, x, class_embedding):
        out = self.bn(x)
        gamma = self.gamma_proj(class_embedding).view(-1, self.num_features, 1, 1)
        beta = self.beta_proj(class_embedding).view(-1, self.num_features, 1, 1)
        return out * gamma + beta

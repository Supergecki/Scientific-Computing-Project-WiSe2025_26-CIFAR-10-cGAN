import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.nz = config['model']['latent_dim']
        self.n_embed = config['model']['embed_dim']
        self.n_classes = config['data']['n_classes']
        self.label_emb = nn.Embedding(self.n_classes, self.n_embed)

# project and reshape the input noise and label embedding to a feature map of size (256, 4, 4)
        self.project = nn.Sequential(
            nn.Linear(self.nz + self.n_embed, 256*4*4),
            nn.BatchNorm1d(256*4*4),
            nn.ReLU(True)
        )

# convolutional layers to upsample the feature maps
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer 2: 256x4x4 -> 128x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Layer 3: 128x8x8 -> 64x16x16
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output values in the range [-1, 1]
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.project(x)
        x = x.view(-1, 256, 4, 4)  # Reshape to (batch_size, 256, 4, 4)
        return self.net(x)
    
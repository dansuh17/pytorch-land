"""
Implementations of models for EBGAN.

Zhao et al - "Energy Based Generative Adversarial Network" (2016)
See : https://arxiv.org/abs/1609.03126
"""
import torch
from torch import nn


class EBGANDiscriminator(nn.Module):
    """
    EBGAN Discriminator model for training MNIST.
    """
    def __init__(self):
        super().__init__()
        # input : (b x 1 x 28 x 28)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=4, stride=2, padding=1, bias=False),  # out: (b x 64 x 14 x 14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (b x 128 x 8 x 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (b x 256 x 3 x 3)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (b x 128 x 6 x 6)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, bias=False),  # (b x 64 x 14 x 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # (b x 1 x 28 x 28)
            nn.Tanh()
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 1)


class EBGANGenerator(nn.Module):
    """
    EBGAN Generator model for training MNIST.
    The number of conv layers follows the 'best model' specified in Appendix C of the paper.

    - nLayerG = 5
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        # input : (b x latent_dim x 1 x 1)
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # out : (b x 64 x 2 x 2)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=64,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # (b x 32 x 4 x 4)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # (b x 16 x 8 x 8)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, 4, 2, 2, bias=False),  # (b x 8 x 14 x 14)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 1, 4, 2, 1, bias=False),  # (b x 1 x 28 x 28)
            nn.Tanh(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        # the input size is given as : (b x latent_dim).
        # Transform this to (b x latent_dim x 1 x 1)
        return self.net(x.view(-1, self.latent_dim, 1, 1))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 1)


if __name__ == '__main__':
    dummy_input = torch.randn(10, 1, 28, 28)  # MNIST dimensions
    D = EBGANDiscriminator()
    disc = D(dummy_input)
    print(disc.size())

    latent_size = 100
    dummy_latent = torch.randn(10, latent_size)
    G = EBGANGenerator(latent_dim=latent_size)
    gen = G(dummy_latent)
    print(gen.size())

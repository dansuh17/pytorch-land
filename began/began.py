import torch
from torch import nn


class BEGANGenerator(nn.Module):
    """Generator module for Boundary-Equilibrium GAN for training on MNIST dataset."""
    def __init__(self, latent_dim: int):
        super().__init__()
        # share the architecture with decoder module of discriminator
        self.linear = nn.Linear(
            in_features=latent_dim, out_features=7 * 7 * 24)  # out : (b x (7 x 7 x 24))
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=24, out_channels=24,
                kernel_size=3, stride=1, padding=1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=24, out_channels=24,
                kernel_size=3, stride=1, padding=1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),  # (b x 24 x 14 x 14)

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 14 x 14)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 14 x 14)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),  # (b x 24 x 28 x 28)

            nn.Conv2d(24, 1, 3, 1, 1, bias=False),  # (b x 1 x 28 x 28)
            nn.Tanh(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.linear(x).view(-1, 24, 7, 7)
        return self.net(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 1)


class BEGANDiscriminiator(nn.Module):
    """Discriminator module for Boundary-Equilibrium GAN for training MNIST dataset"""
    def __init__(self):
        super().__init__()
        self.latent_dim = 100

        # encoder network
        self.encoder = nn.Sequential(
            # input size : (b x 1 x 28 x 28)
            nn.Conv2d(
                in_channels=1, out_channels=8,
                kernel_size=3, padding=1, bias=False),  # (b x 8 x 28 x 28)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 3, 1, 1, bias=False),  # (b x 16 x 28 x 28)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (b x 16 x 14 x 14)

            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),  # (b x 16 x 14 x 14)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 24, 3, 1, 1, bias=False),  # (b x 24 x 14 x 14)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2, 1),  # (b x 24 x 7 x 7)

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # latent variable created through a linear layer
        self.encoder_linear = nn.Linear(
            in_features=7 * 7 * 24, out_features=self.latent_dim)  # (b x latent_dim)

        # decoder network
        self.decoder_linear = nn.Linear(
            in_features=self.latent_dim, out_features=7 * 7 * 24)  # (b x (24 x 7 x 7))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=24, out_channels=24,
                kernel_size=3, stride=1, padding=1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=24, out_channels=24,
                kernel_size=3, stride=1, padding=1, bias=False),  # (b x 24 x 7 x 7)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),  # (b x 24 x 14 x 14)

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 14 x 14)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(24, 24, 3, 1, 1, bias=False),  # (b x 24 x 14 x 14)
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2, inplace=True),

            nn.UpsamplingNearest2d(scale_factor=2),  # (b x 24 x 28 x 28)

            nn.Conv2d(24, 1, 3, 1, 1, bias=False),  # (b x 1 x 28 x 28)
            nn.Tanh(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.encoder(x)
        h = self.encoder_linear(x.view(-1, 7 * 7 * 24))
        x = self.decoder_linear(h)
        x = self.decoder(x.view(-1, 24, 7, 7))
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0, 1)


if __name__ == '__main__':
    dummy_input = torch.randn((10, 1, 28, 28))

    z = torch.randn((10, 100))
    G = BEGANGenerator(latent_dim=100)
    gen = G(z)
    print(gen.size())

    D = BEGANDiscriminiator()
    encoded = D(dummy_input)
    print(encoded.size())

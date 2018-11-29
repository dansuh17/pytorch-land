"""
Implementation of f-GAN.
"""
import torch
from torch import nn


class FGanGenerator(nn.Module):
    """Generator model of f-GAN."""
    def __init__(self, input_dim: int):
        super().__init__()
        # create model with a stack of 5 convolutional layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_dim, out_channels=1024, kernel_size=4,
                stride=1, padding=0, bias=False),  # output : (1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),  # (3, 64, 64)
            nn.Tanh()
        )

        self.apply(self.init_weights)

    def forward(self, z):
        return self.net(z)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


class FGanDiscriminator(nn.Module):
    """Discriminator model of f-GAN."""
    def __init__(self, activation_func: nn.Module):
        super().__init__()
        self.net = nn.Sequential(
            # input : (3, 64, 64)
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False),  # output: (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # output : (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # output : (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # (1, 1, 1)
        )
        self.activation_func = activation_func  # output activation function

        self.apply(self.init_weights)

    def forward(self, x):
        return self.activation_func(self.net(x)).view(-1, 1)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # artificially create 4D tensor (representing a batch of 1D vector having size 100)
    z = torch.randn((3, 100, 1, 1))
    g = FGanGenerator(100)
    print(g)
    output = g(z)
    print(output.size())

    d = FGanDiscriminator(activation_func=nn.Sigmoid())
    print(d)
    disc_out = d(output)
    print(disc_out)
    print(disc_out.size())

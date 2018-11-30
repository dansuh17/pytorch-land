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
        self.net = nn.Sequential(  # input : (input_dim, 1, 1)
            nn.ConvTranspose2d(
                in_channels=input_dim, out_channels=512, kernel_size=3,
                stride=1, padding=0, bias=False),  # output : (1024, 3, 3)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 3, 2, 0, bias=False),  # (512, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (256, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),  # (128, 28, 28)
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
            # input : (1, 28, 28)
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3,
                stride=2, padding=1, bias=False),  # output: (64, 28, 28)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),  # output : (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # output : (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, 2, 2, bias=False),  # (512, 3, 3)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 3, 1, 0, bias=False),  # (1, 1, 1)
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
    from .divergence import GanDivergence, KLDivergence
    # artificially create 4D tensor (representing a batch of 1D vector having size 100)
    z = torch.randn((3, 100, 1, 1))
    g = FGanGenerator(100)
    print(g)
    output = g(z)
    print(output.size())

    d = FGanDiscriminator(activation_func=GanDivergence().output_activation())
    print(d)
    disc_out = d(output)
    print(disc_out)
    print(disc_out.size())

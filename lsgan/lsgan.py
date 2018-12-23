import torch
from torch import nn


class LSGanGenerator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.linear_out = 7 * 7 * 256

        self.linear_layer = nn.Sequential(
            nn.Linear(
                in_features=noise_dim, out_features=self.linear_out),
            nn.BatchNorm1d(self.linear_out),
        )
        self.conv_layers = nn.Sequential(

        )

    def forward(self, x):
        pass


class LSGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    noise_dim = 100
    noise = torch.randn((10, noise_dim))
    print('Generator output size')
    generator = LSGanGenerator(noise_dim)
    gen = generator(noise)
    print(gen.size())

    print('Discriminator output size')
    discriminator = LSGanDiscriminator()
    disc = discriminator(gen)
    print(disc.size())

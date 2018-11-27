from functools import reduce
import torch
from torch import nn


class InfoGanMnistGenerator(nn.Module):
    """Generator model for InfoGAN."""
    def __init__(self):
        super().__init__()
        self.input_dim = 74  # 10 categorical, 2 continuous, 62 noise
        # create model with 5 stacks of linear layer
        self.linear_part = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(in_features=1024, out_features=6272),  # out : (b, 128, 7, 7)
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(6272),
        )
        self.conv_part = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(self.init_weights)

    def forward(self, z):
        out = self.linear_part(z)
        return self.conv_part(out.view(-1, 128, 7, 7))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class InfoGanMnistDiscriminator(nn.Module):
    """Discriminator model for InfoGAN.
    This also shares layer with the model that learns the proxy distribution Q(c|x).

    The last layers after the `linear_part` returns
    separate values D(x) and Q(c|x), respectively.
    """
    def __init__(self, img_size: tuple, noise_size: int, code_size: int):
        super().__init__()
        self.numel = reduce(lambda x, y: x * y, img_size)
        self.noise_size = noise_size
        self.code_size = code_size
        # shared layers for D and Q
        self.conv_part = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
        )
        self.linear_part = nn.Sequential(
            nn.Linear(in_features=(128 * 7 * 7), out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer for discriminator : D(x)
        self.discriminator_front = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        # layer for latent code reproduction distribution : Q(c|x)
        self.code_front = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.code_size)
        )

        self.apply(self.init_weights)

    def forward(self, in_data):
        x = self.conv_part(in_data)
        x = self.linear_part(x.view(-1, 128 * 7 * 7))
        dx = self.discriminator_front(x)  # discriminated results : D(x)
        cx = self.code_front(x)  # latent code reproduction : Q(c|x)
        return dx, cx

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    noise_size = 62
    code_size = 12
    latent_code_size = 74
    zc = torch.randn((10, latent_code_size))
    g = InfoGanMnistGenerator()
    print(g)
    output = g(zc)
    print(output.size())

    d = InfoGanMnistDiscriminator(
        img_size=(1, 28, 28), noise_size=noise_size, code_size=code_size)
    print(d)
    disc_out, latent_code_repr = d(output)
    print(disc_out.size())
    print(latent_code_repr.size())

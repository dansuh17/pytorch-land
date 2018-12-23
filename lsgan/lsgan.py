import torch
from torch import nn


class LSGanGenerator(nn.Module):
    def __init__(self, noise_dim: int):
        super().__init__()
        self.linear_out = 7 * 7 * 256
        self.conv_input = (256, 7, 7)

        self.linear_layer = nn.Sequential(
            nn.Linear(
                in_features=noise_dim, out_features=self.linear_out),  # out : (b, self.linear_out)
            nn.BatchNorm1d(self.linear_out),
            nn.ReLU(inplace=True),
        )
        # transform the input into : (b, 256, 7, 7)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256,
                kernel_size=3, stride=2, padding=2, bias=False),  # (b, 256, 11, 11)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, 1, padding=2, bias=False),  # (b, 256, 9, 9)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, 2, padding=3, bias=False),  # (b, 256, 13, 13)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, 3, 1, padding=2, bias=False),  # (b, 256, 11, 11)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 3, 2, padding=3, bias=False),  # (b, 256, 17, 17)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, 2, padding=2, bias=False),  # (b, 256, 31, 31)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, padding=0, bias=False),  # (b, 3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear_layer(x)
        x = x.view(-1, *self.conv_input)
        return self.conv_layers(x)


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

    # print('Discriminator output size')
    # discriminator = LSGanDiscriminator()
    # disc = discriminator(gen)
    # print(disc.size())

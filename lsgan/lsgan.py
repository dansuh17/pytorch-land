import torch
from torch import nn


class LSGanGenerator(nn.Module):
    """Least Squares GAN generator module."""
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

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)


class LSGanDiscriminator(nn.Module):
    """Least Squares GAN discriminator module."""
    def __init__(self):
        super().__init__()
        # in : (b, 3, 64, 64)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=5, stride=2, padding=2, bias=False),  # (b, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 128, 5, 2, padding=2, bias=False),  # (b, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.Conv2d(128, 256, 5, 2, padding=2, bias=False),  # (b, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 512, 5, 2, padding=2, bias=False),  # (b, 512, 4, 4)
        )
        self.conv_output_size = 512 * 4 * 4
        self.linear = nn.Linear(in_features=self.conv_output_size, out_features=1)

    def forward(self, x):
        x = self.net(x)
        return self.linear(x.view(-1, self.conv_output_size))

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


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
    print(disc)
    print(disc.size())

    loss = nn.MSELoss()
    valid = torch.zeros((10, 1))
    print(loss(disc, valid) * 0.5)
    print(loss(disc, torch.ones((10, 1))) * 0.5 + loss(disc, valid) * 0.5)

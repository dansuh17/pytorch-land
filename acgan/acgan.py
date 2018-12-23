import torch
from torch import nn


class ACGanGenerator(nn.Module):
    """ACGAN generator model assuming CIFAR-10 image generation."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim  # inputs expected to have size (input_dim, 1, 1)
        # transpose conv size : (input - 1) * stride - 2 * padding + 1
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_dim, out_channels=512,
                kernel_size=4, stride=1, padding=0, bias=False),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),  # (3, 32, 32)
            nn.Tanh()
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)


class ACGanDiscriminator(nn.Module):
    """ACGAN discriminator model assuming CIFAR-10 image inputs."""
    def __init__(self, num_class: int):
        super().__init__()
        self.num_class = num_class
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128,
                kernel_size=4, stride=2, padding=1, bias=False),  # out : (128, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),

            nn.Conv2d(256, 512, 4, 2, 1),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output_numel = 512 * 4 * 4

        # separate linear layer for classification and discrimination
        self.fc_classifier = nn.Linear(self.output_numel, num_class)
        self.fc_discriminator = nn.Linear(self.output_numel, 1)

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        conv_out = self.net(x)
        conv_out = conv_out.view(-1, self.output_numel)  # common convolutional layers
        return (
            torch.sigmoid(self.fc_discriminator(conv_out)),  # discrimination output
            torch.log_softmax(self.fc_classifier(conv_out), dim=1),  # classification output
        )

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    dummy_input = torch.randn((10, 256, 1, 1))
    G = ACGanGenerator(input_dim=256)
    D = ACGanDiscriminator(num_class=10)

    gen = G(dummy_input)
    print('Generated data size.')
    print(gen.size())

    disc, classified = D(gen)
    print('Discriminator output sizes.')
    print(disc.size())
    print(classified.size())

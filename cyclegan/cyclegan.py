import torch
from torch import nn


class CycleGanGenerator(nn.Module):
    """
    CycleGAN Generator module that uses 6 residual blocks,
    assuming (3 x 128 x 128)-sized input images.

    The network with 6 residual blocks consists of:
    c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3
    """
    def __init__(self):
        super().__init__()
        # input: (3, 128, 128)
        # output: (3, 128, 128)
        # TODO: try different types of padding layers - reflection / replication
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False),  # out: (b, 64, 128, 128)
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # add downsampling layers
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # (b, 128, 64, 64)
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),  # (b, 256, 32, 32)
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # add 6 residual blocks
            ResBlock(256),  # (b, 256, 32, 32)
            ResBlock(256),  # (b, 256, 32, 32)
            ResBlock(256),  # (b, 256, 32, 32)
            ResBlock(256),  # (b, 256, 32, 32)
            ResBlock(256),  # (b, 256, 32, 32)
            ResBlock(256),  # (b, 256, 32, 32)

            # add upsampling layers
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),  # (b, 128, 64, 64)
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, bias=False),
            nn.InstanceNorm2d(64),  # (b, 64, 128, 128)
            nn.ReLU(inplace=True),

            # final layer
            nn.Conv2d(64, 3, 7, padding=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class CycleGanDiscriminator(nn.Module):
    """
    CycleGAN Discriminator module.
    Implements a 70 x 70 PatchGAN.

    "
    Let Ck denote a 4x4 convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
    ...

    The discriminator architecture is:
    C64, C128, C256, C512
    "

    The number "70" comes by tracing back the receptive field size that will result as 1 x 1 output.
    That is, the convolutional network here is mathematically equivalent to
    "chopp(ing) up the image into 70 x 70 overlapping patches" and mapping each patch to a 1 x 1 output.

    The equation goes: receptive = (output_size - 1) * stride + kernel_size

    See Also:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L532
        https://github.com/phillipi/pix2pix/blob/master/scripts/receptive_field_sizes.m

    (1, 1) -> (4, 4) -> (7, 7) -> (16, 16) -> (34, 34) -> (70, 70)
    """
    def __init__(self):
        super().__init__()
        # input: (3, 128, 128)
        self.net = nn.Sequential(
            # patch: (70 x 70)
            nn.Conv2d(
                in_channels=3, out_channels=64,
                kernel_size=4, stride=2, padding=1, bias=False),  # out: (b, 64, 64, 64)
            # "We do not use InstanceNorm for the first C64 layer"
            nn.LeakyReLU(0.2, inplace=True),

            # patch: (34 x 34)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (b, 128, 32, 32)
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # patch: (16 x 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (b, 256, 16, 16)
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # patch: (7 x 7)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (b, 512, 8, 8)
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # patch: (4 x 4)
            nn.Conv2d(512, 256, 4, 1, 1, bias=False),  # (b, 256, 7, 7)
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # patch: (1 x 1)
            nn.Conv2d(256, 1, 4, 1, 1, bias=False),  # (b, 1, 6, 6)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """
    Residual block proposed by ResNet, but here it uses InstanceNorm instead of batch normalization.

    See Also: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, chan: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(chan),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(chan, chan, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(chan),
        )

    def forward(self, x):
        """
        No ReLU layer at the end of Residual Block - it shows slightly better performance.

        See Also:
            http://torch.ch/blog/2016/02/04/resnets.html
        """
        return x + self.net(x)


if __name__ == '__main__':
    block = ResBlock(3)
    z = torch.randn((10, 3, 128, 128))
    print(block(z).size())

    g = CycleGanGenerator()
    print(g(z).size())

    d = CycleGanDiscriminator()
    print(d(z).size())

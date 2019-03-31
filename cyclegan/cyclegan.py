import torch
from torch import nn
import torch.nn.functional as F


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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, bias=False),  # out: (b, 64, 128, 128)
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
    def __init__(self):
        super().__init__()
        # input: (3, w, h)
        # output: 70 by 70 patchGANs

    def forward(self, x):
        pass


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
        # TODO: no ReLU layer at the end of Residual Block - it shows slightly better performance.
        # TODO: See also: http://torch.ch/blog/2016/02/04/resnets.html
        return F.relu(x + self.net(x), inplace=True)


if __name__ == '__main__':
    block = ResBlock(3)
    z = torch.randn((10, 3, 128, 128))
    print(block(z).size())

    g = CycleGanGenerator()
    print(g(z).size())

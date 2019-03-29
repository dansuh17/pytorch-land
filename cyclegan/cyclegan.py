import torch
from torch import nn
import torch.nn.functional as F


class CycleGanGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (3, w, h) (3, 256, 256) for monet dataset
        # output: (3, w, h) (3, 256, 256) for monet dataset

    def forward(self, x):
        pass


class CycleGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # input: (3, w, h)
        # output: 70 by 70 patchGANs

    def forward(self, x):
        pass


class ResBlock(nn.Module):
    """
    Residual block proposed by ResNet.

    See Also: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, chan: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=chan, out_channels=chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(chan, chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chan),
        )

    def forward(self, x):
        # TODO: no ReLU layer at the end of Residual Block - it shows slightly better performance.
        # TODO: See also: http://torch.ch/blog/2016/02/04/resnets.html
        return F.relu(x + self.net(x), inplace=True)


if __name__ == '__main__':
    block = ResBlock(10)
    z = torch.randn((10, 10, 128, 128))
    print(block(z).size())

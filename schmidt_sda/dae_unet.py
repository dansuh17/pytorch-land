import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resample_required = in_channels != out_channels

        conv_layers = []
        conv_layers.extend([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        for _ in range(2):
            conv_layers.extend([
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        if self.resample_required:
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsampler = None

    def forward(self, x):
        if self.resample_required:
            x = self.conv_layers(x) + self.downsampler(x)
        else:
            x = self.conv_layers(x) + x
        return F.relu(x, inplace=True)


# resnet based denoiser
class DansuhDenoisingCNN(nn.Module):
    """
    My own denoising neural network.
    """
    def __init__(self):
        super().__init__()
        self.resblocks = nn.Sequential(
            ResBlock(in_channels=1, out_channels=4),
            ResBlock(4, 16),
            ResBlock(16, 16),
            ResBlock(16, 64),
            ResBlock(64, 128),
            ResBlock(128, 256),
            ResBlock(256, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            ResBlock(128, 64),
            ResBlock(64, 16),
            ResBlock(16, 1),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.resblocks(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)  # He init (model is using ReLU)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = DansuhDenoisingCNN()
    print(net)
    dummy_input = torch.randn((1, 1, 40, 40))
    out = net(dummy_input)
    print(out.size())

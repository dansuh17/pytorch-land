import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, no_last_relu=False):
        super().__init__()
        self.resample_required = in_channels != out_channels
        self.no_last_relu = no_last_relu

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

        # resample layer - used if the channels are downsized
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
        # ReLU at the last layer may not be required for the final layer for generation
        return x if self.no_last_relu else F.relu(x, inplace=True)


# resnet + U-Net based denoiser
class DansuhDenoisingCNN(nn.Module):
    """
    My own denoising neural network.
    """
    def __init__(self):
        super().__init__()
        self.res_up1 = ResBlock(in_channels=1, out_channels=4, no_last_relu=True)
        self.res_up2 = ResBlock(4, 16)
        self.res_up3 = ResBlock(16, 16)
        self.res_up4 = ResBlock(16, 64)
        self.res_up5 = ResBlock(64, 64)
        self.res_up6 = ResBlock(64, 128)
        self.res_up7 = ResBlock(128, 128)
        self.res_up8 = ResBlock(128, 256)
        self.res_up9 = ResBlock(256, 256)
        self.res_up10 = ResBlock(256, 512)
        self.res_zenith = ResBlock(512, 512)
        self.res_down10 = ResBlock(512, 512)
        self.res_down9 = ResBlock(512, 256)
        self.res_down8 = ResBlock(256, 256)
        self.res_down7 = ResBlock(256, 128)
        self.res_down6 = ResBlock(128, 128)
        self.res_down5 = ResBlock(128, 64)
        self.res_down4 = ResBlock(64, 64)
        self.res_down3 = ResBlock(64, 16)
        self.res_down2 = ResBlock(16, 16)
        self.res_down1 = ResBlock(16, 1, no_last_relu=True)

        self.apply(self.init_weights)

    def forward(self, x):
        up1 = self.res_up1(x)
        up2 = self.res_up2(up1)
        up3 = self.res_up3(up2)
        up4 = self.res_up4(up3)
        up5 = self.res_up5(up4)
        up6 = self.res_up6(up5)
        up7 = self.res_up7(up6)
        up8 = self.res_up8(up7)
        up9 = self.res_up9(up8)
        up10 = self.res_up10(up9)
        out = self.res_zenith(up10)
        out = self.res_down10(out) + up10
        out = self.res_down9(out) + up9
        out = self.res_down8(out) + up8
        out = self.res_down7(out) + up7
        out = self.res_down6(out) + up6
        out = self.res_down5(out) + up5
        out = self.res_down4(out) + up4
        out = self.res_down3(out) + up3
        out = self.res_down2(out) + up2
        out = self.res_down1(out)
        return out

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

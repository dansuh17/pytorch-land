import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Define a residual block.
    Residual block takes an input, passes through a number of 2d convolutional layers,
    and then returns the sum of the output of convolutional layers and original input (the residual).
    """
    def __init__(self, in_channels, out_channels, convs=2):
        super().__init__()
        self.downsample_required = in_channels != out_channels
        if self.downsample_required:
            conv_layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
            conv_layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(convs - 1)])
            self.downsamp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            conv_layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                for _ in range(convs)]
            self.downsamp_conv = None  # no need to downsample the residual with 1x1 conv
        # make conv layers a sequential operation
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        if self.downsample_required:
            # residual should be downsampled in dimension + increased in features to match the size
            x = self.conv_layers(x) + self.downsamp_conv(x)
        else:
            x = self.conv_layers(x) + x
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes: int, mode='A'):
        super().__init__()
        # input : (b x 3 x 224 x 224)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)  # output: (b x 64 x 112 x 112)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (b x 64 x 56 x 56)
        # create residual blocks
        res_1 = [ResBlock(64, 64, convs=2) for _ in range(3)]  # (b x 64 x 56 x 56)
        res_2 = [ResBlock(64, 128, convs=2)] + [ResBlock(128, 128, convs=2) for _ in range(3)]  # (b x 128 x 28 x 28)
        res_3 = [ResBlock(128, 256, convs=2)] + [ResBlock(256, 256, convs=2) for _ in range(5)]  # (b x 256 x 14 x 14)
        res_4 = [ResBlock(256, 512, convs=2)] + [ResBlock(512, 512, convs=2) for _ in range(2)]  # (b x 512 x 7 x 7)

        # make residual blocks a sequential operation
        self.res = nn.Sequential(*(res_1 + res_2 + res_3 + res_4))
        self.avg_pool = nn.AvgPool2d(kernel_size=3)  # (b x 512 x 2 x 2)
        self.fc = nn.Linear(512 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.maxpool(self.conv_1(x))
        x = self.avg_pool(self.res(x))
        print(x.size())
        x = x.view(-1, 512 * 2 * 2)
        return self.fc(x)


if __name__ == '__main__':
    residual = ResBlock(128, 256, convs=2)
    print(residual)
    samp = torch.randn((10, 128, 28, 28))
    print(residual(samp).size())

    net = ResNet34(1000)
    print(net)
    sample_batch = torch.randn((10, 3, 224, 224))
    print(net(sample_batch).size())

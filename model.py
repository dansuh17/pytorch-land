import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """Initialize weights for convolutional and linear layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


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
            for _ in range(convs - 1):
                conv_layers.extend([nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1)])
            self.downsamp_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            conv_layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
            for _ in range(convs - 1):
                conv_layers.extend([nn.ReLU(inplace=True), nn.Conv2d(in_channels, out_channels, 3, padding=1)])
            self.downsamp_conv = None  # no need to downsample the residual with 1x1 conv
        # make conv layers a sequential operation
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        if self.downsample_required:
            # residual should be downsampled in dimension + increased in features to match the size
            x = self.conv_layers(x) + self.downsamp_conv(x)
        else:
            x = self.conv_layers(x) + x
        return F.relu(x, inplace=True)


class ResNet32(nn.Module):
    def __init__(self, num_classes: int, input_dim: int):
        super().__init__()
        # input size : (b x 3 x 32 x 32)
        dim_shrink_rate = 1
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # (b x 16 x 32 x 32)
        res1 = [ResBlock(16, 16, convs=2) for _ in range(5)]  # (b x 16 x 32 x 32)
        res2 = [ResBlock(16, 32, convs=2)] + [ResBlock(32, 32, convs=2) for _ in range(4)]  # (b x 32 x 16 x 16)
        dim_shrink_rate *= 2
        res3 = [ResBlock(32, 64, convs=2)] + [ResBlock(64, 64, convs=2) for _ in range(4)]  # (b x 64 x 8 x 8)
        dim_shrink_rate *= 2

        self.res = nn.Sequential(*(res1 + res2 + res3))
        self.avg_pool = nn.AvgPool2d(kernel_size=2)  # (b x 64 x 4 x 4)
        dim_shrink_rate *= 2
        self.feature_dim = input_dim // dim_shrink_rate
        self.fc = nn.Linear(64 * self.feature_dim * self.feature_dim, num_classes)

        self.apply(init_weights)  # initielize weights for all submodules

    def forward(self, x):
        x = self.conv_1(x)
        x = self.avg_pool(self.res(x))
        x = x.view(-1, 64 * self.feature_dim * self.feature_dim)
        return self.fc(x)

class ResNet34(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, mode='A'):
        super().__init__()
        dim_shrink_rate = 1  # no shrink -- initially
        # input : (b x 3 x 224 x 224)
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)  # output: (b x 64 x 112 x 112)
        dim_shrink_rate *= 2

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (b x 64 x 56 x 56)
        dim_shrink_rate *= 2

        # create residual blocks
        res_1 = [ResBlock(64, 64, convs=2) for _ in range(3)]  # (b x 64 x 56 x 56)
        res_2 = [ResBlock(64, 128, convs=2)] + [ResBlock(128, 128, convs=2) for _ in range(3)]  # (b x 128 x 28 x 28)
        dim_shrink_rate *= 2
        res_3 = [ResBlock(128, 256, convs=2)] + [ResBlock(256, 256, convs=2) for _ in range(5)]  # (b x 256 x 14 x 14)
        dim_shrink_rate *= 2
        res_4 = [ResBlock(256, 512, convs=2)] + [ResBlock(512, 512, convs=2) for _ in range(2)]  # (b x 512 x 7 x 7)
        dim_shrink_rate *= 2

        # make residual blocks a sequential operation
        self.res = nn.Sequential(*(res_1 + res_2 + res_3 + res_4))
        self.avg_pool = nn.AvgPool2d(kernel_size=3)  # (b x 512 x 2 x 2)
        dim_shrink_rate *= 3
        self.feature_dim = input_dim // dim_shrink_rate

        self.fc = nn.Linear(512 * self.feature_dim * self.feature_dim, num_classes)
        self.apply(init_weights)  # initialize all submodules' weights

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

    net = ResNet34(1000, input_dim=224)
    print(net)
    sample_batch = torch.randn((10, 3, 224, 224))
    print(net(sample_batch).size())

    # test for cifar-10
    net2 = ResNet32(10, input_dim=32)
    print(net2)
    sample_batch = torch.randn((10, 3, 32, 32))
    print(net2(sample_batch).size())

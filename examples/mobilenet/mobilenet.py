import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                kernel_size=3, stride=stride, padding=padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # input size : (b, 3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)  # (b, 32, 112, 112)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.dw_conv1 = DepthwiseSeparableConv2d(in_channels=32, out_channels=64)  # (b, 64, 112, 112)
        self.dw_conv2 = DepthwiseSeparableConv2d(64, 128, stride=2)  # (b, 128, 56, 56)
        self.dw_conv3 = DepthwiseSeparableConv2d(128, 128)  # (b, 128, 56, 56)
        self.dw_conv4 = DepthwiseSeparableConv2d(128, 256, stride=2)  # (b, 256, 28, 28)
        self.dw_conv5 = DepthwiseSeparableConv2d(256, 256)  # (b, 256, 28, 28)
        self.dw_conv6 = DepthwiseSeparableConv2d(256, 512, stride=2)  # (b, 512, 14, 14)
        # increase model capacity on this dimension
        self.dw_conv7 = DepthwiseSeparableConv2d(512, 512)  # (b, 512, 14, 14)
        self.dw_conv8 = DepthwiseSeparableConv2d(512, 512)  # (b, 512, 14, 14)
        self.dw_conv9 = DepthwiseSeparableConv2d(512, 512)  # (b, 512, 14, 14)
        self.dw_conv10 = DepthwiseSeparableConv2d(512, 512)  # (b, 512, 14, 14)
        self.dw_conv11 = DepthwiseSeparableConv2d(512, 512)  # (b, 512, 14, 14)
        self.dw_conv12 = DepthwiseSeparableConv2d(512, 1024, stride=2)  # (b, 1024, 7, 7)
        self.dw_conv13 = DepthwiseSeparableConv2d(1024, 1024, stride=2, padding=4)  # (b, 1024, 7, 7)
        # TODO: calculate kernel size
        self.avg_pool = nn.AvgPool2d(kernel_size=7)  # (b, 1024, 1, 1)
        self.fc = nn.Linear(1024, self.num_classes)  # (b, 1000)

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        x = self.dw_conv6(x)
        x = self.dw_conv7(x)
        x = self.dw_conv8(x)
        x = self.dw_conv9(x)
        x = self.dw_conv10(x)
        x = self.dw_conv11(x)
        x = self.dw_conv12(x)
        x = self.dw_conv13(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # test with expected input size
    net = MobileNet(1000)
    dummy_input = torch.randn((10, 3, 224, 224))
    output = net(dummy_input)
    print(net)
    print(output.size())

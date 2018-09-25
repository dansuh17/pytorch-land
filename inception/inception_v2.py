import torch
from torch import nn


class BasicConvBlock(nn.Module):
    """Basic unit of convolutional layer that consists of convolution,
    batch normalization, and ReLU nonlinear activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionA(nn.Module):
    """Factorizes 5x5 to two 3x3 layers"""
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1 = BasicConvBlock(in_channels, out_channels=64, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConvBlock(in_channels, 48, 1),
            BasicConvBlock(48, 64, 3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
            BasicConvBlock(in_channels, pool_features, 1),
        )

        self.branch4 = nn.Sequential(
            BasicConvBlock(in_channels, 64, 1),
            BasicConvBlock(64, 96, 3, padding=1),
            BasicConvBlock(96, 96, 3, padding=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionDimRedA(nn.Module):
    """Reduce the dimension using 1x1 conv and pooling."""
    def __init__(self, in_channels):
        super().__init__()
        # example input size : (b, 288, 35, 35)
        self.branch1 = BasicConvBlock(in_channels, 384, 3, stride=2)  # (b, 384, 17, 17)
        self.branch2 = nn.Sequential(
            BasicConvBlock(in_channels, 64, kernel_size=1),  # (b, 64, 35, 35)
            BasicConvBlock(64, 96, kernel_size=3, padding=1),  # (b, 96, 35, 35)
            BasicConvBlock(96, 96, kernel_size=3, stride=2),  # (b, 96, 17, 17)
        )
        self.pool_branch = nn.MaxPool2d(kernel_size=3, stride=2)  # (b, 288, 17, 17)
        # after concat : (b, 768, 17, 17)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        pooled = self.pool_branch(x)
        return torch.cat([branch1, branch2, pooled], 1)


class InceptionB(nn.Module):
    """Fatctorizes 7x7 into a number of 7x1 convolution layers."""
    def __init__(self, in_channels, factorized_channels):
        super().__init__()
        # example input size : (b, 768, 17, 17)
        self.branch1 = BasicConvBlock(in_channels, 192, kernel_size=1)  # (b, 192, 17, 17)
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),  # (b, 768, 17, 17)
            BasicConvBlock(in_channels, 192, kernel_size=1),  # (b, 192, 17, 17)
        )
        self.branch3 = nn.Sequential(
            BasicConvBlock(in_channels, factorized_channels, kernel_size=1),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, factorized_channels, kernel_size=(1, 7), padding=(0, 3)),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, 192, kernel_size=(7, 1), padding=(3, 0)),  # (b, 192, 17, 17)
        )
        self.branch4 = nn.Sequential(
            BasicConvBlock(in_channels, factorized_channels, kernel_size=1),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, factorized_channels, kernel_size=(7, 1), padding=(3, 0)),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, factorized_channels, kernel_size=(1, 7), padding=(0, 3)),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, factorized_channels, kernel_size=(7, 1), padding=(3, 0)),  # (b, f_ch, 17, 17)
            BasicConvBlock(factorized_channels, 192, kernel_size=(1, 7), padding=(0, 3)),  # (b, 192, 17, 17)
        )
        # after concatenation : (b, 768, 17, 17)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionDimRedB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            BasicConvBlock(in_channels, 192, kernel_size=1),
            BasicConvBlock(192, 320, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConvBlock(in_channels, 192, kernel_size=1),
            BasicConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(192, 192, kernel_size=3, stride=2),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        pooled = self.pool(x)
        return torch.cat([b1, b2, pooled], 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = BasicConvBlock(in_channels, out_channels=320, kernel_size=1)

        self.branch2 = BasicConvBlock(in_channels, 384, 1)
        self.branch2_a = BasicConvBlock(384, 384, (1, 3), padding=(0, 1))
        self.branch2_b = BasicConvBlock(384, 384, (3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(
            BasicConvBlock(in_channels, 448, 1),
            BasicConvBlock(448, 384, 3, padding=1),
        )
        self.branch3_a = BasicConvBlock(384, 384, (1, 3), padding=(0, 1))
        self.branch3_b = BasicConvBlock(384, 384, (3, 1), padding=(1, 0))

        self.branch_pool = BasicConvBlock(in_channels, 192, 1)

    def forward(self, x):
        b1 = self.branch1(x)

        b2_pre = self.branch2(x)
        b2_a = self.branch2_a(b2_pre)
        b2_b = self.branch2_b(b2_pre)
        b2 = torch.cat([b2_a, b2_b], 1)

        b3_pre = self.branch3(x)
        b3_a = self.branch3_a(b3_pre)
        b3_b = self.branch3_b(b3_pre)
        b3 = torch.cat([b3_a, b3_b], 1)

        pooled = self.branch_pool(x)
        return torch.cat([b1, b2, b3, pooled], 1)


class InceptionV2(nn.Module):
    """Main Inception v.2 Network"""
    def __init__(self, num_classes=1000, img_dim=299):
        super().__init__()
        dim = img_dim
        if img_dim > 80:  # for large input sizes - following original paper implementation
            # input : (b, 3, 299, 299)
            self.stem = nn.Sequential(
                BasicConvBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2),  # (b, 32, 149, 149)
                BasicConvBlock(32, 32, 3),  # (b, 32, 147, 147)
                BasicConvBlock(32, 64, 3, padding=1),  # (b, 64, 147, 147)
                nn.MaxPool2d(kernel_size=3, stride=2),  # (b, 64, 73, 73)
                BasicConvBlock(64, 80, 1),  # (b, 80, 73, 73)
                BasicConvBlock(80, 192, 3),  # (b, 192, 71, 71)
                nn.MaxPool2d(kernel_size=3, stride=2),  # (b, 192, 35, 35)
            )
            # calculate output dimension
            dim = self.dim_reduction(dim, 3, 2)
            dim = self.dim_reduction(dim, 3, padding=0)
            dim = self.dim_reduction(dim, 3, padding=1)
            dim = self.dim_reduction(dim, 3, stride=2)  # pool
            dim = self.dim_reduction(dim, 3, padding=0)
            dim = self.dim_reduction(dim, 3, stride=2)
        else:  # for smaller input sizes
            self.stem = nn.Sequential(
                BasicConvBlock(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                BasicConvBlock(32, 96, 3, padding=1),
                BasicConvBlock(96, 192, 3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            # calculate output dimension
            dim = self.dim_reduction(dim, 3, stride=2)  # pool

        self.inception_a = nn.Sequential(
            InceptionA(192, pool_features=32),  # (b, 256, 35, 35)
            InceptionA(256, pool_features=64),  # (b, 288, 35, 35)
            InceptionA(288, pool_features=64),  # (b, 288, 35, 35)
        )
        self.dim_reduced = InceptionDimRedA(in_channels=288)  # (b, 768, 17, 17)
        dim = self.dim_reduction(dim, 3, stride=2)

        self.inception_b = nn.Sequential(
            InceptionB(768, factorized_channels=128),
            InceptionB(768, factorized_channels=160),
            InceptionB(768, factorized_channels=160),
            InceptionB(768, factorized_channels=192),
        )  # (b, 768, 17, 17)
        self.dim_reduced2 = InceptionDimRedB(in_channels=768)  # (b, 1280, 8, 8)
        dim = self.dim_reduction(dim, 3, stride=2)

        self.inception_c = nn.Sequential(
            InceptionC(1280),  # (b, 2048, 8, 8)
            InceptionC(2048),  # (b, 2048, 8, 8)
            nn.AvgPool2d(kernel_size=dim),  # (b, 2048, 1, 1)
            nn.Dropout(p=0.5, inplace=True),
        )

        self.fc = nn.Linear(2048, num_classes)  # (b, num_classes)

    @staticmethod
    def dim_reduction(dim_size, kernel_size, stride=1, padding=0):
        return (dim_size + 2 * padding - kernel_size) // stride + 1

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.dim_reduced(x)
        x = self.inception_b(x)
        x = self.dim_reduced2(x)
        x = self.inception_c(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    sample = torch.randn((10, 192, 35, 35))
    inc_a = InceptionA(192, 32)
    print(inc_a(sample).size())

    sample2 = torch.randn((10, 3, 299, 299))
    inc = InceptionV2(num_classes=1000, img_dim=299)
    print(inc(sample2).size())

    sample2 = torch.randn((10, 3, 64, 64))
    inc = InceptionV2(num_classes=10, img_dim=64)
    print(inc(sample2).size())

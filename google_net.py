import torch
from torch import nn


class InceptionA(nn.Module):
    """Fatctorizes 5x5 to two 3x3 layers"""
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 1),
            nn.Conv2d(48, 64, 3, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels, pool_features, 1),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionDimRedA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 384, 3, stride=2)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, stride=2),
        )
        self.pool_branch = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        pooled = self.pool_branch(x)
        return torch.cat([branch1, branch2, pooled], 1)


class InceptionB(nn.Module):
    """Fatctorizes 7x7"""
    def __init__(self, in_channels, factorized_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 192, kernel_size=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, factorized_channels, kernel_size=1),
            nn.Conv2d(factorized_channels, factorized_channels, kernel_size=(1, 7), padding=(0, 3))
            nn.Conv2d(factorized_channels, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, factorized_channels, kernel_size=1),
            nn.Conv2d(factorized_channels, factorized_channels, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(factorized_channels, factorized_channels, kernel_size=(1, 7), padding=(0, 3))
            nn.Conv2d(factorized_channels, factorized_channels, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(factorized_channels, 192, kernel_size=(1, 7), padding=(0, 3))
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionV2(nn.Module):
    def __init__(self):
        super().__init__()
        # input : (b, 3, 299, 299)
        self.pre_inception = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),  # (b, 32, 149, 149)
            nn.Conv2d(32, 32, 3),  # (b, 32, 147, 147)
            nn.Conv2d(32, 64, 3, padding=1),  # (b, 64, 147, 147)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b, 64, 73, 73)
            nn.Conv2d(64, 80, 1),  # (b, 80, 73, 73)
            nn.Conv2d(80, 192, 3),  # (b, 192, 71, 71)
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b, 192, 35, 35)
        )
        self.inception_a = nn.Sequential(
            InceptionA(192, pool_features=32),  # (b, 256, 35, 35)
            InceptionA(256, pool_features=64),  # (b, 288, 35, 35)
            InceptionA(288, pool_features=64),  # (b, 288, 35, 35)
        )
        self.dim_reduced = InceptionDimRedA(in_channels=288)  # (b, 768, 17, 17)

    def forward(self, x):
        x = self.pre_inception(x)
        print(x.size())
        x = self.inception_a(x)
        print(x.size())
        return self.dim_reduced(x)


if __name__ == '__main__':
    sample = torch.randn((10, 192, 35, 35))
    inc_a = InceptionA(192, 32)
    # print(inc_a(sample).size())

    sample2 = torch.randn((10, 3, 299, 299))
    inc = InceptionV2()
    print(inc(sample2).size())

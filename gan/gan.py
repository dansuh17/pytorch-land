from functools import reduce
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_dim: int, img_size: tuple):
        super().__init__()
        self.img_size = img_size
        numel = reduce(lambda x, y: x * y, img_size)
        # create model with 5 stacks of linear layer
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, numel),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), self.img_size)  # resize to image
        return x


class Discriminator(nn.Module):
    def __init__(self, img_size: tuple):
        super().__init__()
        numel = reduce(lambda x, y: x * y, img_size)
        self.net = nn.Sequential(
            nn.Linear(in_features=numel, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


if __name__ == '__main__':
    z = torch.randn((3, 100))
    g = Generator(input_dim=100, img_size=(1, 28, 28))
    print(g)
    output = g(z)
    print(output.size())

    d = Discriminator(img_size=(1, 28, 28))
    print(d)
    disc_out = d(output)
    print(disc_out.size())

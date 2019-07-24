import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=784, out_features=10)
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.net(x)


if __name__ == '__main__':
    dummy = torch.randn((10, 1, 28, 28))
    m = SimpleModel()
    out = m(dummy)

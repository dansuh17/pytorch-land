"""
"""
import torch
from torch import nn


class WganGpGenerator(nn.Module):
    """Generator model of WGAN-GP.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # create model with a stack of 5 convolutional layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_dim, out_channels=1024, kernel_size=4,
                stride=1, padding=0, bias=False),  # output : (1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),  # (3, 64, 64)
            nn.Tanh(),
        )

        self.apply(self.init_weights)

    def forward(self, z):
        return self.net(z)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


class WganGpDiscriminator(nn.Module):
    """Discriminator model of WGAN-GP.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input : (3, 64, 64)
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False),  # output: (64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # output : (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # output : (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # (1, 1, 1)
        )

        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x).view(-1, 1)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # artificially create 4D tensor (representing a batch of 1D vector having size 100)
    z = torch.randn((3, 100, 1, 1))
    g = WganGpGenerator(100)
    print(g)
    output = g(z)
    print(output.size())

    d = WganGpDiscriminator()
    print(d)
    disc_out = d(output)
    print(disc_out)
    print(disc_out.size())
    print(disc_out.squeeze().size())
    disc_out.backward(torch.ones(disc_out.size()))

    # testing calculation of GP
    import random
    alpha = random.random()
    real_img = torch.randn((3, 3, 64, 64))
    img_interp = (real_img - output) * alpha + output
    img_interp = img_interp.detach().requires_grad_()  # set requires_grad=True to store the grad value
    score_img_interp = d(img_interp)
    score_img_interp.backward(torch.ones((3, 1)))

    grads = img_interp.grad
    grad_norm = grads.view((3, -1)).norm(dim=1)
    print(grads.size())
    print(grad_norm.size())

    grad_pen = 10 * torch.pow(grad_norm - 1, 2)
    # out = output.detach().requires_grad_()
    # disc_out2 = d(out)
    # disc_out2.backward(torch.ones(disc_out2.size()))
    # print(out.grad)
    print(grad_pen)
    print(score_img_interp.size())
    res = score_img_interp.squeeze() + grad_pen
    res.backward(torch.ones((3, )))

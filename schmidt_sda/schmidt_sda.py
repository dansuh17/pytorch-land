import torch
from torch import nn


class SchmidtSDA(nn.Module):
    """
    Stacked convolutional denoising autoencoder implementation.

    See Also: Schmidt et al. "Stacked Denoising and Stacked Convolutional Autoencoders" (2017)
    """
    def __init__(self, input_channel: int, input_height: int, input_width: int):
        super().__init__()
        self.input_width = input_width
        self.input_height = input_height

        # default pooling layer configuration
        self.pool_kernel_size = 2
        self.pool_kernel_stride = 2

        # calculate the dimensions after the pooling layer
        self.after_pool_width = (input_width - self.pool_kernel_size) // self.pool_kernel_stride + 1
        self.after_pool_height = (input_height - self.pool_kernel_size) // self.pool_kernel_stride + 1
        self.after_pool_channels = 16

        self.linear_in = \
            self.after_pool_width * self.after_pool_height * self.after_pool_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=self.pool_kernel_size,
                stride=self.pool_kernel_stride,
                return_indices=True),  # useful when unpooling
        )
        # fc part of encoder
        self.encoder_comp = nn.Sequential(
            nn.Linear(in_features=self.linear_in, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 16),
        )
        self.decoder_comp = nn.Sequential(
            nn.Linear(16, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.linear_in),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2)
        # fc part of decoder
        self.decoder_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4, 1, 3, padding=1),
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        x, pool_indices = self.encoder_conv(x)  # latent variable
        # the latent vector that compresses information
        z = self.encoder_comp(x.view(-1, self.linear_in))
        x = self.decoder_comp(z)
        # specify output size when unpooling
        x = self.unpool(
            x.view(-1,
                   self.after_pool_channels,
                   self.after_pool_height,
                   self.after_pool_width),
            pool_indices,
            output_size=(-1,
                         self.after_pool_channels,
                         self.input_height,
                         self.input_width))
        return self.decoder_conv(x), z

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)  # He init (model is using ReLU)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = SchmidtSDA(1, 32, 32)
    dummy_sample = torch.randn((10, 1, 32, 32))
    out, z = net(dummy_sample)
    print('Z size')
    print(z.size())
    print('Output size')
    print(out.size())

    net = SchmidtSDA(1, 40, 11)
    dummy_sample = torch.randn((10, 1, 40, 11))
    out, z = net(dummy_sample)
    print('Z size')
    print(z.size())
    print('Output size')
    print(out.size())
    # for name, param in net.named_parameters():
    #     print(name)
    #     print(param)

import torch
from torch import nn
import torch.nn.functional as F


class LayerBlockUpsample(nn.Module):
    """
    Layer block used for upsampling in the Generator model.
    Consists of two convolutional layers with upsampling layer.
    Upsampling is done through replication.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, neg_slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # upsample by replication
        return self.net(x)


class LayerBlockDownSample(nn.Module):
    """
    Layer block used for downsampling in the Discriminator model.
    Consists of two convolutional layers with downsampling layer by average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, neg_slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2)  # downsample
        return self.net(x)


class ToRGB(nn.Module):
    """
    To_RGB layer.
    This layer is used to convert the output features at the end of Generator
    to RGB image tensor, having three output channels.
    """
    def __init__(self, in_channel: int, dim: int):
        super().__init__()
        self.dim = dim
        self.conv_layer = nn.Conv2d(
            in_channels=in_channel, out_channels=3, kernel_size=1, bias=False)
        self.linear_features = 3 * dim * dim
        self.linear_layer = nn.Linear(
            in_features=self.linear_features, out_features=self.linear_features)

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(-1, self.linear_features)
        # reshape output to desired image tensor
        return self.linear_layer(out).view(-1, 3, self.dim, self.dim)

    @staticmethod
    def init_weights(m: nn.Module):
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class FromRGB(nn.Module):
    """
    From_RGB layer.
    This layer is used to convert input image tensor provided to the Discriminator
    into a tensor having certain number of features,
    in order to feed into the convolutional network.
    """
    def __init__(self, out_channel: int, neg_slope=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channel, kernel_size=1, bias=False),
            nn.LeakyReLU(neg_slope, inplace=True),
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m: nn.Module):
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ProGanGenerator(nn.Module):
    """
    1024 x 1024 Generator model for "Progressive Growing of GANs" paper.
    """
    def __init__(self):
        super().__init__()
        self.first_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=512, kernel_size=4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # define input keyword arguments required for to_RGB layer at each level
        self.to_rgb_args = (
            {'in_channel': 512, 'dim': 4},
            {'in_channel': 512, 'dim': 8},
            {'in_channel': 512, 'dim': 16},
            {'in_channel': 512, 'dim': 32},
            {'in_channel': 256, 'dim': 64},
            {'in_channel': 128, 'dim': 128},
            {'in_channel': 64, 'dim': 256},
            {'in_channel': 32, 'dim': 512},
            {'in_channel': 16, 'dim': 1024},
        )
        self.curr_level = 0
        self.to_rgb_layer = ToRGB(**self.to_rgb_args[self.curr_level])

        self.net = nn.ModuleList([
            self.first_block,  # out: (b, 512, 4, 4)
            LayerBlockUpsample(in_channels=512, out_channels=512),  # (b, 512, 8, 8)
            LayerBlockUpsample(512, 512),  # (b, 512, 16, 16)
            LayerBlockUpsample(512, 512),  # (b, 512, 32, 32)
            LayerBlockUpsample(512, 256),  # (b, 256, 64, 64)
            LayerBlockUpsample(256, 128),  # (b, 128, 128, 128)
            LayerBlockUpsample(128, 64),  # (b, 64, 256, 256)
            LayerBlockUpsample(64, 32),  # (b, 32, 512, 512)
            LayerBlockUpsample(32, 16),  # (b, 16, 1024, 1024)
        ])

        self.num_layers = len(self.net)
        assert self.num_layers == len(self.to_rgb_args)

        # initialize weights
        self.apply(self.weight_init)

    def reset_to_rgb_layer(self, level: int):
        if level != self.curr_level:
            self.curr_level = level
            self.to_rgb_layer = ToRGB(**self.to_rgb_args[self.curr_level])
            print(f'To RGB layer created for level: {level}')

    def forward(self, *inpt):
        """
        Forward pass through the generator.
        The input should not only contain the latent vector but also
        the progression level.
        Higher progression level means output images of higher resolution.

        Args:
            *inpt(nn.Tensor, int): tuple of latent vector batch and progression level

        Returns:
            generated image tensor
        """
        input_latent, level = inpt
        self.reset_to_rgb_layer(level)

        out = input_latent
        for layer in self.net[:level + 1]:
            out = layer(out)
        return self.to_rgb_layer(out)

    @staticmethod
    def weight_init(m: nn.Module):
        """
        Section 4.1 of Karras et al.,
        "We ... use a trivial N(0, 1) initialization
        and then explicitly scale the weights at runtime.
        """
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ProGanDiscriminator(nn.Module):
    """
    1024 x 1024 Discriminator model for "Progressive Growing of GANs" paper.
    """
    def __init__(self):
        super().__init__()
        self.last_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),  # out: (b, 512, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, bias=False),  # (b, 512, 1, 1)
            nn.LeakyReLU(0.2, inplace=True),
        )

        # define output channels required by from_RGB layers
        self.from_rgb_out_channels = (16, 32, 64, 128, 256, 512, 512, 512, 512)
        self.curr_level = 0

        self.net = nn.ModuleList([
            LayerBlockDownSample(in_channels=16, out_channels=32),  # out: (b, 32, 512, 512)
            LayerBlockDownSample(32, 64),  # (b, 64, 256, 256)
            LayerBlockDownSample(64, 128),  # (b, 128, 128, 128)
            LayerBlockDownSample(128, 256),  # (b, 256, 64, 64)
            LayerBlockDownSample(256, 512),  # (b, 512, 32, 32)
            LayerBlockDownSample(512, 512),  # (b, 512, 16, 16)
            LayerBlockDownSample(512, 512),  # (b, 512, 8, 8)
            LayerBlockDownSample(512, 512),  # (b, 512, 4, 4)
            self.last_block,  # (b, 512, 1, 1)
        ])
        self.linear_layer = nn.Linear(in_features=512, out_features=1)

        self.num_layers = len(self.net)
        assert self.num_layers == len(self.from_rgb_out_channels)

        self.from_rgb_layer = FromRGB(
            self.from_rgb_out_channels[self.num_layers - 1 - self.curr_level])

        # initialize weights
        self.apply(self.weight_init)

    def reset_from_rgb_layer(self, level: int):
        if self.curr_level != level:
            self.curr_level = level
            self.from_rgb_layer = FromRGB(
                self.from_rgb_out_channels[self.num_layers - 1 - self.curr_level])
            print(f'From RGB layer created for level: {level}')

    def forward(self, *inpt):
        """
        Forward-pass through the discriminator.
        The input should not only contain images but also the progression level.
        The progression level indicates the resolution of the image being generated.

        Args:
            *inpt(nn.Tensor, int): tuple of image batch and progression level

        Returns:
            critic tensor
        """
        input_img, level = inpt
        self.reset_from_rgb_layer(level)

        out = self.from_rgb_layer(input_img)
        # start index for the layer
        start_idx = self.num_layers - 1 - level
        for layer in self.net[start_idx:]:
            out = layer(out)
        out = out.view(-1, 512)
        return self.linear_layer(out).view(-1, 1, 1, 1)

    @staticmethod
    def weight_init(m: nn.Module):
        """
        Section 4.1 of Karras et al.,
        "We ... use a trivial N(0, 1) initialization
        and then explicitly scale the weights at runtime.
        """
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    rand_dim = 512
    batch_size = 1

    progan_g = ProGanGenerator()
    print(progan_g)

    progan_d = ProGanDiscriminator()
    print(progan_d)

    for level in range(9):
        dummy_rand = torch.randn((batch_size, rand_dim, 1, 1))
        g_out = progan_g(dummy_rand, level)
        print(f'Generator output for level: {level} = {g_out.size()}')
        d_out = progan_d(g_out, level)
        assert d_out.size() == torch.Size([batch_size, 1, 1, 1])

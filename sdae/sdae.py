import torch
from torch import nn


class SDAE(nn.Module):
    """A simple 3-layer stacked denoising autoencoder (denoted "SdA-3" in the literature),
    proposed by Vincent et al. (2008).

    This implementation assumes training with MNIST dataset only.
    (See `noisy_dataset.NoisyMnistDataset` about how dataset looks like.)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.SELU(inplace=True),
            nn.Linear(400, 200),
            nn.SELU(inplace=True),
            nn.Linear(200, 100),
            nn.SELU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.SELU(inplace=True),
            nn.Linear(200, 400),
            nn.SELU(inplace=True),
            nn.Linear(400, input_dim),
        )
        self.init_weights(self)

    def forward(self, x):
        in_size = x.size()
        x = x.view(in_size[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(in_size)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # use kaiming due to the use of SELU nonlinearity
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


class GeneralizedSDAE(nn.Module):
    """Generalized N-stacked denoising autoencoder
    that can have arbitrary number of stacked linear layers
    and arbitrary input sizes.
    """
    def __init__(self, input_dim: int, feature_size=100, num_stacks=3):
        super().__init__()
        # feature_size should be smaller, unless it has sparse condition, which is not implemented here
        assert input_dim > feature_size
        dim_reduction_size = (input_dim - feature_size) // num_stacks
        # compute the intermediate feature sizes
        feature_sizes = list(map(
            lambda idx: input_dim - dim_reduction_size * (idx + 1),
            range(num_stacks)))
        feature_sizes = [input_dim] + feature_sizes
        feature_sizes[-1] = feature_size
        self.input_dim = input_dim
        self.feature_size = feature_size

        # build an encoder
        encoder_layers = []
        for i in range(num_stacks):
            in_features = feature_sizes[i]
            out_features = feature_sizes[i + 1]

            encoder_layers.append(nn.Linear(in_features, out_features))
            encoder_layers.append(nn.BatchNorm1d(out_features))
            encoder_layers.append(nn.SELU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        # build the decoder
        decoder_layers = []
        for i in range(num_stacks):
            # going backwards
            in_features = feature_sizes[num_stacks - i]
            out_features = feature_sizes[num_stacks - 1 - i]

            decoder_layers.append(nn.Linear(in_features, out_features))
            decoder_layers.append(nn.BatchNorm1d(out_features))
            # don't SELU the last activation
            if i != num_stacks - 1:
                decoder_layers.append(nn.SELU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        in_size = x.size()
        x = x.view(in_size[0], -1)
        x = self.encoder(x)
        assert x.size(1) == self.feature_size
        x = self.decoder(x)
        return x.view(in_size)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # use kaiming due to the use of SELU nonlinearity
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # test with dummy input
    input_size = 40 * 11
    net = SDAE(input_dim=input_size)
    dummy_input = torch.randn((10, 1, input_size))
    output = net(dummy_input)
    print(output.size())
    assert output.size()[2] == input_size

    # test generalized SDAE
    input_size = 575
    net = GeneralizedSDAE(input_dim=input_size, num_stacks=5)
    print(net)
    dummy_input = torch.randn((10, 1, input_size))
    output = net(dummy_input)
    print(output.size())

    print(net.__class__.__name__)

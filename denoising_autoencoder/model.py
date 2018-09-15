from torch import nn


class SDAE(nn.Module):
    """A simple 3-layer stacked denoising autoencoder (denoted "SdA-3" in the literature),
    proposed by Vincent et al. (2008).

    This implementation assumes training with MNIST dataset only.
    (See `noisy_dataset.NoisyMnistDataset` about how dataset looks like.)
    """
    def __init__(self, input_dim):
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # use kaiming due to the use of SELU nonlinearity
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

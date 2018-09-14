from torch import nn


class SDAE(nn.Module):
    """3-layer stacked denoising autoencoder (denoted by SdA-3 in the literature),
    proposed by Vincent et al. (2008).
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

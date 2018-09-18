from torch import nn


class SCDAE(nn.Module):
    """
    Stacked Convolutional Denoising Autoencoder.
    """
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=1000, kernel_size=11)
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
        conv2 = nn.Conv2d(in_channels=?, out_channels=1500, kernel_size=2)

    def forward(self, x):
        # TODO: whitening done at the dataset?
        pass

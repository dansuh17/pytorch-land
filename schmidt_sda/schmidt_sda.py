from torch import nn


class SchmidtSDA(nn.Module):
    """
    Schmid et al. "Stacked Denoising and Stacked Convolutional Autoencoders" (2017)
    """
    def __init__(self, input_channel: int, input_dim: int):
        super().__init__()
        after_pool = (input_dim - 1) // 2 + 1
        self.encoder = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Linear(in_features=(after_pool * after_pool * 16), out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 16),
        )
        self.decoder = nn.Sequential(
            # TODO:
        )
        self.init_weights(self)

    def forward(self, x):
        z = self.encoder(x)  # latent variable
        x_restored = self.decoder(x)
        return z, x_restored

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

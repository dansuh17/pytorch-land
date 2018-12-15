from torch import nn


class UnrolledGanGenerator(nn.Module):
    """Generator model for Unrolled GAN.

    This setup is for training 2D mixture of Gaussians toy dataset
    explained in section 3.1 and appendix A in
    Metz et al. - "Unrolled Generative Adversarial Nets" (2017)
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.hidden_size = 128
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, output_size),
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)


class UnrolledGanDiscriminator(nn.Module):
    """Discriminator model for Unrolled GAN."""
    def __init__(self, input_size: int):
        super().__init__()
        self.hidden_size = 128
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )

        # initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

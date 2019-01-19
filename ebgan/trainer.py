from .ebgan import EBGANGenerator, EBGANDiscriminator
from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer


class EBGANTrainer(NetworkTrainer):
    """
    Trainer for Auxiliary Classifier GANs (AC-GAN).
    """
    def __init__(self, config: dict):
        print('Configuration')
        print(config)

        self.input_dim = config['input_dim']
        self.batch_size = config['batch_size']
        self.total_epoch = config['epoch']
        self.display_imgs = config['display_imgs']
        self.height = config['height']
        self.width = config['width']

        # create data loader maker
        loader_maker = MNISTLoaderMaker(
            data_root='data_in', batch_size=self.batch_size, naive_normalization=True)
        self.num_class = loader_maker.num_classes

        generator = EBGANGenerator(input_dim=self.input_dim)
        discriminator = EBGANDiscriminator(num_class=self.num_class)
        models = {
            'ACGan_G': ModelInfo(
                model=generator,
                input_size=(self.input_dim, 1, 1),
                metric='loss_g',
                comparison=operator.lt
            ),
            'ACGan_D': ModelInfo(
                model=discriminator,
                input_size=(3, self.height, self.width),
                metric='loss_d',
            ),
        }

        # create criteria
        criteria = {
            'g_criteria': nn.BCELoss(),
            'd_criteria': nn.BCELoss(),
            'classification_loss': nn.NLLLoss(),  # assumes log-softmaxed values from discriminator
        }

        # create optimizers
        self.lr_init = config['lr_init']
        optimizers = {
            'optimizer_g': optim.Adam(
                generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
            'optimizer_d': optim.Adam(
                discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
        }

        # create the trainer instance
        super().__init__(
            models, loader_maker, criteria, optimizers, epoch=self.total_epoch)

        self.epoch = 0

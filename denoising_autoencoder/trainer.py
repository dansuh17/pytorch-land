import os
import torch
import torch.nn as nn
import torch.optim as optim
from ..base_trainer import NetworkTrainer
from .model import SDAE
from ..dataset import load_mnist
from tensorboardX import SummaryWriter


class SDAETrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.input_root_dir = 'sdae_data_in'
        self.output_root_dir = 'sdae_data_out'
        self.log_dir = os.path.join(self.output_root_dir, 'tblogs')
        self.models_dir = os.path.join(self.output_root_dir, 'models')
        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.batch_size = 128
        self.num_devices = 4
        self.lr_init = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ids = list(range(self.num_devices))
        self.seed = torch.initial_seed()
        print('Using random seed : {}'.format(self.seed))

        self.train_dataloader, self.val_dataloader, self.test_dataloader, misc = \
            load_mnist(self.input_root_dir, self.batch_size)
        self.image_dim = misc['image_dim']
        print('Dataloader created')

        sdae = SDAE(input_dim=self.image_dim * self.image_dim).to(self.device)
        self.sdae = torch.nn.parallel.DataParallel(sdae, device_ids=self.device_ids)
        print('Model created')
        print(self.sdae)

        self.optimizer = optim.Adam(params=self.sdae.parameters(), lr=self.lr_init)
        print('Optimizer created')

        self.summ_writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer created')

        self.criterion = nn.MSELoss(reduction='elementwise_mean')
        print('Criterion : {}'.format(self.criterion))

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', verbose=True, factor=0.1)
        print('LR scheduler created')

        self.epoch = 0
        self.total_steps = 0
        print('Starting from - epoch : {}, step: {}'.format(self.epoch, self.total_steps))

    def train(self):
        for _ in range(self.epoch, self.end_epoch):
            self.summ_writer.add_scalar('epoch', self.epoch, self.total_steps)
            # train
            for imgs, targets in self.train_dataloader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)

                out = self.sdae(imgs)

            # validate

            self.lr_scheduler.step(val_loss)
            self.epoch += 1
        # test
        self.test()

    def test(self):
        pass

    def validate(self):
        pass

    def save_model(self, filename: str):
        pass

    def save_checkpoint(self, filename: str):
        pass

    def cleanup(self):
        self.summ_writer.close()

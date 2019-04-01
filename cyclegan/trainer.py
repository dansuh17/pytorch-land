import operator
from typing import Dict
from torch import nn
from torch import optim
from cyclegan.cyclegan import CycleGanDiscriminator, CycleGanGenerator
from datasets.img_transfer import Monet2PhotoLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage


class CycleGANTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        print('Configuration: ')
        print(config)

        self.batch_size = config['batch_size']
        self.data_root_dir = config['data_root']
        self.total_epoch = config['epoch']
        self.lr = config['lr']
        self.num_devices = 4

        loader_maker = Monet2PhotoLoaderMaker(
            self.batch_size, self.data_root_dir, downsize_half=True)
        img_size = (3, 128, 128)

        g = CycleGanGenerator()
        f = CycleGanGenerator()
        d_x = CycleGanDiscriminator()
        d_y = CycleGanDiscriminator()

        models = {
            'CycleGAN_G': ModelInfo(
                model=g,
                input_size=img_size,
                metric='loss_g',
                comparison=operator.lt,
            ),
            'CycleGAN_F': ModelInfo(
                model=f,
                input_size=img_size,
                metric='loss_f',
            ),
            'CycleGAN_Dx': ModelInfo(
                model=d_x,
                input_size=img_size,
                metric='loss_d_x',
            ),
            'CycleGAN_Dy': ModelInfo(
                model=d_y,
                input_size=img_size,
                metric='loss_d_y',
            ),
        }

        criteria = {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss(),
        }

        optimizers = {
            'optimizer_g': optim.Adam(g.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_f': optim.Adam(f.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_d_x': optim.Adam(d_x.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_d_y': optim.Adam(d_y.parameters(), self.lr, betas=(0.5, 0.999)),
        }

        super().__init__(
            models, loader_maker, criteria, optimizers,
            epoch=self.total_epoch, num_devices=self.num_devices, lr_scheduler=None)

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        pass


if __name__ == '__main__':
    import json
    with open('began/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = CycleGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

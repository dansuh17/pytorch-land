import torch
from torch import nn
from torch import optim
from torchland.trainer import NetworkTrainer, TrainStage
from torchland.datasets.img_popular import MNISTLoaderBuilder
from .model import SimpleModel


class VGGTrainer(NetworkTrainer):

    def __init__(self, config: dict):
        super().__init__(epoch=100)
        self.lr_init = 3e-4
        self.batch_size = 64

        vgg = SimpleModel()
        self.add_model(name='vgg', model=vgg, input_size=(1, 28, 28), metric='loss')
        self.set_dataloader_builder(
            MNISTLoaderBuilder(data_root='data_out', batch_size=self.batch_size))
        self.add_criterion('cross_entropy', nn.CrossEntropyLoss())
        self.add_optimizer('vgg_optim', optim.Adam(params=vgg.parameters(), lr=self.lr_init))

    def run_step(self, models, criteria, optimizers,
                 input_, train_stage, *args, **kwargs):
        out = models.vgg.model(input_[0])
        loss = criteria.cross_entropy(out, input_[1])

        if train_stage == TrainStage.TRAIN:
            optimizers.vgg_optim.zero_grad()
            loss.backward()
            optimizers.vgg_optim.step()

        return out, loss


if __name__ == '__main__':
    t = VGGTrainer({})
    t.fit()
    t.cleanup()

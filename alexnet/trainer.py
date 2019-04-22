"""
Implementation of training of alexnet.
"""
import operator
from typing import Dict
import torch
from torch import optim, nn
from torch.optim.optimizer import Optimizer
from .alexnet import AlexNet
from base_trainer import NetworkTrainer, ModelInfo, TrainStage
from datasets.img_popular import ImageNetLoaderMaker


class AlexNetTrainer(NetworkTrainer):
    """
    Trainer for AlexNet.
    """
    def __init__(self, config: dict):
        print('Configuration')
        print(config)

        self.batch_size = config['batch_size']
        self.total_epoch = config['epoch']
        self.data_root = config['data_root']
        self.lr = config['lr']
        self.num_devices = config['num_devices']
        self.img_dim = 227

        loadermaker = ImageNetLoaderMaker(self.data_root, self.batch_size, num_workers=4, img_dim=self.img_dim)
        self.input_size = (3, self.img_dim, self.img_dim)

        alexnet = AlexNet()
        models = {
            'alexnet': ModelInfo(
                model=alexnet, input_size=self.input_size, metric='loss', comparison=operator.lt),
        }
        print('Model created:')
        print(alexnet)

        criteria = {'cross_entropy': nn.CrossEntropyLoss()}

        optimizers = {'optim': optim.Adam(alexnet.parameters(), self.lr, betas=(0.5, 0.999))}

        lr_scheduler = (optim.lr_scheduler.StepLR(optimizers['optim'], step_size=20, gamma=0.5), )

        super().__init__(
            models, loadermaker, criteria, optimizers,
            epoch=self.total_epoch, num_devices=self.num_devices, lr_scheduler=lr_scheduler)

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria: Dict[str, nn.modules.loss._Loss],
                 optimizer: Dict[str, Optimizer],
                 input_: torch.Tensor,
                 train_stage: TrainStage,
                 *args, **kwargs):
        # parse inputs
        img_batch, target_batch = input_

        alexnet = model['alexnet'].model
        cross_entropy_loss = criteria['cross_entropy']
        opt = optimizer['optim']

        # forward pass
        out_features = alexnet(img_batch)

        # calculate loss
        loss = cross_entropy_loss(out_features, target_batch)

        # update parameters
        if train_stage == TrainStage.TRAIN:
            opt.zero_grad()
            loss.backward()
            opt.step()

        outputs = (out_features, target_batch)
        losses = (loss, )
        return outputs, losses

    @staticmethod
    def make_performance_metric(input_: torch.Tensor, output, loss) -> dict:
        out_features, target_batch = output
        _, predictions = torch.max(out_features, 1)
        accuracy = torch.sum(predictions == target_batch).float() / target_batch.size()[0]

        return {
            'loss': loss[0].item(),
            'accuracy': accuracy.item(),
        }


if __name__ == '__main__':
    import json
    with open('alexnet/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = AlexNetTrainer(config)
    trainer.fit()
    trainer.cleanup()

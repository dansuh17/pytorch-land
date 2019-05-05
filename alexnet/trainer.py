"""
Implementation of training of AlexNet.

See also: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import operator
from typing import Dict
import torch
from torch import optim, nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import torchvision
from .alexnet import AlexNet
from base_trainer import NetworkTrainer, ModelInfo, TrainStage
from datasets.img_popular import ImageNetLoaderMaker, CIFAR10LoaderMaker


# fix random seeds for experimenting
# torch.manual_seed(1004)
# torch.cuda.manual_seed_all(2019)


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
        self.img_dim = 227  # correct configuration to give values displayed in alexnet paper

        # loadermaker = ImageNetLoaderMaker(
        #     self.data_root, self.batch_size, num_workers=8, img_dim=self.img_dim)
        loadermaker = CIFAR10LoaderMaker(self.data_root, self.batch_size, num_workers=4, img_dim=self.img_dim)  # TODO: debug
        self.input_size = (3, self.img_dim, self.img_dim)

        alexnet = AlexNet()
        models = {
            'alexnet': ModelInfo(
                model=alexnet, input_size=self.input_size, metric='loss', comparison=operator.lt),
        }
        print('Model created:')
        print(alexnet)

        criteria = {'cross_entropy': nn.CrossEntropyLoss()}

        opt = optim.SGD(alexnet.parameters(), self.lr, momentum=0.9, weight_decay=0.0005)
        optimizers = {'optim': opt}

        lr_scheduler = {'steplr': optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)}

        super().__init__(
            models, loadermaker, criteria, optimizers,
            epoch=self.total_epoch, num_devices=self.num_devices, lr_scheduler=lr_scheduler)

    def run_step(
            self,
            model: Dict[str, ModelInfo],
            criteria: Dict[str, _Loss],
            optimizer: Dict[str, Optimizer],
            input_: torch.Tensor,
            train_stage: TrainStage,
            *args, **kwargs):
        # parse inputs
        imgs, targets = input_

        alexnet = model['alexnet'].model
        cross_entropy_loss = criteria['cross_entropy']
        opt = optimizer['optim']

        # forward pass
        out_features = alexnet(imgs)

        # calculate loss
        loss = cross_entropy_loss(out_features, targets)

        # update parameters
        if train_stage == TrainStage.TRAIN:
            opt.zero_grad()
            loss.backward()
            opt.step()

        outputs = (out_features, targets)
        losses = (loss, )
        return outputs, losses

    def pre_fit(self):
        # show an example image before training
        labels = []
        display_imgs = []
        for i in range(20):
            img, target = self.train_dataloader.dataset[i]
            labels.append(str(target))
            display_imgs.append(img)

        grid = torchvision.utils.make_grid(torch.stack(display_imgs), nrow=20, normalize=True)
        self.writer.add_image('example_imgs', grid, self.global_step)
        self.writer.add_text('labels', ', '.join(labels), self.global_step)

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

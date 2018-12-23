from typing import Dict
import torch
from torch import nn
from torch import optim
import torchvision
import operator
from datasets.img_popular import CIFAR10LoaderMaker
from .acgan import ACGanDiscriminator
from .acgan import ACGanGenerator
from base_trainer import NetworkTrainer
from base_trainer import ModelInfo
from base_trainer import TrainStage


class ACGanTrainer(NetworkTrainer):
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
        loader_maker = CIFAR10LoaderMaker(
            data_root='data_in', batch_size=self.batch_size, naive_normalization=True)
        self.num_class = loader_maker.num_classes

        generator = ACGanGenerator(input_dim=self.input_dim)
        discriminator = ACGanDiscriminator(num_class=self.num_class)
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
            'd_criteria': nn.BCELoss(),
            'g_criteria': nn.BCELoss(),
            'classification_loss': nn.NLLLoss(),
        }

        # create optimizers
        self.lr_init = config['lr_init']
        optimizers = {
            'optimizer_d': optim.Adam(
                discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
            'optimizer_g': optim.Adam(
                generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
        }

        # create the trainer instance
        super().__init__(
            models, loader_maker, criteria, optimizers, epoch=self.total_epoch)

        self.epoch = 0

    def make_noise_input(self):
        # noise vector
        z_size = (self.batch_size, self.input_dim - self.num_class, 1, 1)
        classes_size = (self.batch_size, self.num_class)

        # 'random noise' part of the input
        z = torch.randn(z_size)

        # 'class label' part of the input
        classes = torch.zeros(classes_size)
        # randomly generate class labels
        labels = torch.randint(0, self.num_class - 1, (self.batch_size, )).long()

        # index tensor to assign each of randomly generated labels to 1 on the zero matrix
        indices = torch.arange(0, self.batch_size).long()
        classes[indices, labels] = 1

        classes = classes.view(-1, self.num_class, 1, 1)  # make it size (b, 10, 1, 1)
        return torch.cat([z, classes], dim=1).to(self.device), labels.long().to(self.device)

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        """
        Run a single step of AC-GAN training.
        Args:
            model(Dict[str, ModelInfo]): holds model information
            criteria: model criterion
            optimizer: model optimizers
            input_: inputs - will be provided by iterating DataLoader
            train_stage(TrainStage): can be either train, validate, or test

        Returns:
            outputs: any output data that will be used
                after each step of training (e.g. visualization)
            loss: loss values that will be depicted / logged through the training process

        """
        imgs, real_class_targets = input_
        batch_size = imgs.size(0)

        # get models
        generator = model['ACGan_G'].model
        discriminator = model['ACGan_D'].model

        # parse criteria
        d_criteria = criteria['d_criteria']
        g_criteria = criteria['g_criteria']
        classification_criteria = criteria['classification_loss']

        # parse optimizers
        d_optim = optimizer['optimizer_d']
        g_optim = optimizer['optimizer_g']

        # generate input data
        noise_input, fake_class_targets = self.make_noise_input()

        valid = torch.zeros((batch_size, 1)).to(self.device)
        invalid = torch.ones((batch_size, 1)).to(self.device)

        ### train D ###
        generated = generator(noise_input)
        fake_discriminated, fake_classified = discriminator(generated.detach())
        real_discriminated, real_classified = discriminator(imgs)

        # calculate discrimination loss
        fake_disc_loss = d_criteria(fake_discriminated, invalid)
        real_disc_loss = d_criteria(real_discriminated, valid)
        disc_loss = fake_disc_loss + real_disc_loss

        # calculate classification loss
        fake_cls_loss = classification_criteria(fake_classified, fake_class_targets)
        real_cls_loss = classification_criteria(real_classified, real_class_targets)
        cls_loss = fake_cls_loss + real_cls_loss

        # total loss
        d_loss = disc_loss + cls_loss * 10

        # update step against the total loss
        if train_stage == TrainStage.TRAIN:
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        ### train G ###
        noise_input, fake_class_targets = self.make_noise_input()
        generated = generator(noise_input)
        fake_discriminated, fake_classified = discriminator(generated)

        # calculate total loss
        disc_loss = g_criteria(fake_discriminated, valid)
        cls_loss = classification_criteria(fake_classified, fake_class_targets)
        g_loss = disc_loss + cls_loss * 10

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        # collect outputs and losses
        outputs = (
            generated,
            imgs,  # real images
        )
        losses = (d_loss, g_loss, disc_loss, cls_loss)
        return outputs, losses

    @staticmethod
    def make_performance_metric(input_, output, loss):
        """
        Indicate performance metrics to display and compare.
        """
        return {
            'd_loss': loss[0].item(),
            'g_loss': loss[1].item(),
            'disc_loss': loss[2].item(),
            'cls_loss': loss[3].item(),
        }

    def pre_epoch_finish(self, input, output, metric_manager, train_stage: TrainStage):
        """Add example images from validation step just before the end of epoch training."""
        if train_stage == TrainStage.VALIDATE:
            generated_imgs = output[0]
            real_imgs = output[1]
            self.add_generated_image(
                generated_imgs, nrow=self.display_imgs,
                height=self.height,
                width=self.width, name='generated')
            self.add_generated_image(
                real_imgs, nrow=self.display_imgs,
                height=self.height,
                width=self.width, name='real')

    def add_generated_image(self, imgs, nrow, height, width, name: str):
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)


if __name__ == '__main__':
    import json
    with open('acgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = ACGanTrainer(config)
    trainer.fit()
    trainer.cleanup()

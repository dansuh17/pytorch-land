"""
Implementation of Boundary-Equilibrium GAN (2017).
"""
import operator
from typing import Dict
import torch
import torchvision
from torch import nn, optim
from .began import BEGANGenerator, BEGANDiscriminiator
from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage


class BEGANTrainer(NetworkTrainer):
    """
    Trainer for BEGAN.
    """
    def __init__(self, config: dict):
        print('Configuration')
        print(config)

        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.total_epoch = config['epoch']
        self.display_imgs = config['display_imgs']
        self.prop_lr = 0.001  # lambda value in the paper
        self.equilibrium_const = 0.7  # gamma value in the paper
        self.k = 0  # equilibrium regularizing constant init value

        # create data loader maker
        loader_maker = MNISTLoaderMaker(
            data_root='data_in', batch_size=self.batch_size, naive_normalization=True)
        self.img_dim = loader_maker.dim

        generator = BEGANGenerator(latent_dim=self.latent_dim)
        discriminator = BEGANDiscriminiator()

        models = {
            'BEGAN_G': ModelInfo(
                model=generator,
                input_size=(self.latent_dim, ),
                metric='loss_g',
                comparison=operator.lt
            ),
            'BEGAN_D': ModelInfo(
                model=discriminator,
                input_size=(1, self.img_dim, self.img_dim),
                metric='loss_d',
            ),
        }

        # create criteria
        criteria = {'mseloss': nn.MSELoss()}

        # create optimizers
        self.lr_init = config['lr_init']
        optimizers = {
            'optimizer_g': optim.Adam(
                generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
            'optimizer_d': optim.Adam(
                discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
        }

        # define learning rate schedulers
        lr_schedulers = (
            optim.lr_scheduler.StepLR(
                optimizers['optimizer_g'], step_size=20, gamma=0.2),
            optim.lr_scheduler.StepLR(
                optimizers['optimizer_d'], step_size=20, gamma=0.2),
        )

        # create the trainer instance
        super().__init__(
            models, loader_maker, criteria, optimizers,
            epoch=self.total_epoch, num_devices=2, lr_scheduler=lr_schedulers)

        self.epoch = 0

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        imgs, _ = input_

        # parse models
        generator = model['BEGAN_G'].model
        discriminator = model['BEGAN_D'].model

        d_optim = optimizer['optimizer_d']
        g_optim = optimizer['optimizer_g']

        mse_loss = criteria['mseloss']

        ### Train D ###
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)

        gen_imgs = generator(z).detach()
        reconst_real = discriminator(imgs)  # reconstructed img
        reconst_fake = discriminator(gen_imgs)

        # calculate the reconstruction loss of the 'autoencoder D'
        real_loss = mse_loss(reconst_real, imgs)
        fake_loss = mse_loss(reconst_fake, gen_imgs)

        d_loss = real_loss - self.k * fake_loss

        if train_stage == TrainStage.TRAIN:
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        ### Train G ###
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)

        gen_imgs = generator(z)
        reconst_fake_gen = discriminator(gen_imgs)
        fake_loss_gen = mse_loss(reconst_fake_gen, gen_imgs)  # energy assigned to fake images
        g_loss = fake_loss_gen

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        # Update k - equilibrium regularizing constant
        k_update = self.prop_lr * (self.equilibrium_const * real_loss.item() - fake_loss_gen.item())
        self.k = self.k + k_update

        # calculate the convergence measure
        conv_measure = self.convergence_measure(
            real_loss.item(), fake_loss_gen.item(), self.equilibrium_const)

        outputs = (
            gen_imgs,
            imgs,
            reconst_real,
            reconst_fake,
            reconst_fake_gen,
            self.k,
            conv_measure
        )
        loss = (d_loss, g_loss, real_loss, fake_loss)
        return outputs, loss

    def _update_lr(self, val_metrics):
        for lrs in self.lr_schedulers:
            lrs.step()

    @staticmethod
    def convergence_measure(real_loss, gen_fake_loss, equilibrium_const):
        return real_loss + abs(equilibrium_const * real_loss - gen_fake_loss)

    @staticmethod
    def make_performance_metric(input_, output, loss):
        reconst_real = output[2]
        reconst_fake = output[3]
        reconst_fake_gen = output[4]
        k = output[5]
        convergence_measure = output[6]

        d_loss, g_loss, real_loss, fake_loss = loss

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'k': k,
            'reconst_real': torch.mean(reconst_real).item(),
            'reconst_fake': torch.mean(reconst_fake).item(),
            'reconst_fake_gen': torch.mean(reconst_fake_gen).item(),
            'convergence_measure': convergence_measure,
        }

    def pre_epoch_finish(self, input_, output, metric_manager, train_stage: TrainStage):
        if train_stage == TrainStage.VALIDATE:
            gen_img = output[0]
            real_img = output[1]
            self.add_generated_image(
                gen_img, nrow=self.display_imgs, height=None, width=None, name='gen')
            self.add_generated_image(
                real_img, nrow=self.display_imgs, height=None, width=None, name='real')

    def add_generated_image(self, imgs, nrow, height, width, name: str):
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)


if __name__ == '__main__':
    import json
    with open('began/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = BEGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

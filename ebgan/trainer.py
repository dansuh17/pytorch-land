import operator
from typing import Dict
import torch
import torchvision
from torch import nn, optim
from .ebgan import EBGANGenerator, EBGANDiscriminator
from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage


class EBGANTrainer(NetworkTrainer):
    """
    Trainer for Auxiliary Classifier GANs (AC-GAN).
    """
    def __init__(self, config: dict):
        print('Configuration')
        print(config)

        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.total_epoch = config['epoch']
        self.display_imgs = config['display_imgs']
        self.margin = 10  # "the margin is fixed to 10 and not being tuned"

        # create data loader maker
        loader_maker = MNISTLoaderMaker(
            data_root='data_in', batch_size=self.batch_size, naive_normalization=True)
        self.img_dim =loader_maker.dim

        generator = EBGANGenerator(latent_dim=self.latent_dim)
        discriminator = EBGANDiscriminator()

        models = {
            'EBGAN_G': ModelInfo(
                model=generator,
                input_size=(self.latent_dim, ),
                metric='loss_g',
                comparison=operator.lt
            ),
            'EBGAN_D': ModelInfo(
                model=discriminator,
                input_size=(1, self.img_dim, self.img_dim),
                metric='loss_d',
            ),
        }

        # create criteria
        criteria = {
            'l1': nn.L1Loss(),
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

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        imgs, _ = input_

        # parse models
        generator = model['EBGAN_G'].model
        discriminator = model['EBGAN_D'].model

        d_optim = optimizer['optimizer_d']
        g_optim = optimizer['optimizer_g']

        l1_loss = criteria['l1']

        ### Train D
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        margin_tensor = torch.Tensor([self.margin]).to(self.device)

        gen_imgs = generator(z)
        reconst_real = discriminator(imgs)  # reconstructed img
        reconst_fake = discriminator(gen_imgs)

        # calculate the energy assigned == reconstruction loss of the 'autoencoder D'
        energy_real = l1_loss(reconst_real, imgs)
        energy_fake = l1_loss(reconst_fake, gen_imgs)

        zero = torch.zeros([1]).to(self.device)
        d_loss = energy_real + torch.max(margin_tensor - energy_fake, zero)

        if train_stage == TrainStage.TRAIN:
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        ### Train G
        z = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        gen_imgs = generator(z)
        reconst_fake_gen = discriminator(gen_imgs)
        g_loss = l1_loss(reconst_fake_gen, gen_imgs)  # energy assigned to fake images

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        outputs = (gen_imgs, imgs, energy_real, energy_fake)
        loss = (d_loss, g_loss)
        return outputs, loss

    @staticmethod
    def make_performance_metric(input_, output, loss):
        energy_real = output[2]
        energy_fake = output[3]

        d_loss, g_loss = loss

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'energy_real': torch.mean(energy_real).item(),
            'energy_fake': torch.mean(energy_fake).item(),
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
    with open('ebgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = EBGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

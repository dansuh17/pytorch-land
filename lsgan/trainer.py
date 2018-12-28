from typing import Dict
import torch
from torch import nn
from torch import optim
import torchvision
from .lsgan import LSGanDiscriminator
from .lsgan import LSGanGenerator
from datasets.img_popular import LSUNLoaderMaker
from base_trainer import NetworkTrainer
from base_trainer import ModelInfo
from base_trainer import TrainStage


class LSGANTrainer(NetworkTrainer):
    """Trainer for Least-Squares GAN.
    Mao et al. - "Least Squares Generative Adversarial Networks" (2017)
    """
    def __init__(self, config: dict):
        print('Configuration:')
        print(config)

        # parse configs
        self.batch_size = config['batch_size']
        self.input_dim = config['input_dim']
        self.total_epoch = config['epoch']
        self.display_imgs = config['display_imgs']

        # create a loader maker
        loader_maker = LSUNLoaderMaker(data_root='data_in', batch_size=self.batch_size)

        # create models
        generator = LSGanGenerator(noise_dim=self.input_dim)
        discriminator = LSGanDiscriminator()
        models = {
            'LSGan_G': ModelInfo(
                model=generator,
                input_size=(self.input_dim, ),
                metric='loss_g',
            ),
            'LSGan_D': ModelInfo(
                model=discriminator,
                input_size=(self.input_dim, ),
                metric='loss_d',
            )
        }

        # criteria
        criteria = {
            'g_criteria': nn.MSELoss(),
            'd_criteria': nn.MSELoss(),
        }

        # create optimizers
        self.lr_init = config['lr_init']
        optimizers = {
            'g_optim': optim.Adam(
                generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
            'd_optim': optim.Adam(
                discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        }

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
        generator = model['LSGan_G'].model
        discriminator = model['LSGan_D'].model

        # parse criteria and optimizers
        g_criteria, d_criteria = criteria['g_criteria'], criteria['d_criteria']
        g_optim, d_optim = optimizer['g_optim'], optimizer['d_optim']

        valid = torch.zeros((self.batch_size, 1)).to(self.device)
        invalid = torch.ones((self.batch_size, 1)).to(self.device)

        ###############
        ### train D ###
        ###############
        noise_vec = torch.randn((self.batch_size, self.input_dim))

        # loss function - refer to eq(9) on original paper
        disc_true = discriminator(imgs)
        disc_fake = discriminator(generator(noise_vec).detach())

        loss_true = 0.5 * d_criteria(disc_true, valid)  # D wants real imgs to be labeled 1
        loss_fake = 0.5 * d_criteria(disc_fake, invalid)  # D wants generated imgs to be labeled 0
        loss_d = loss_fake + loss_true

        if train_stage == TrainStage.TRAIN:
            d_optim.zero_grad()
            loss_d.backward()
            d_optim.step()

        ###############
        ### train G ###
        ###############
        noise_vec = torch.randn((self.batch_size, self.input_dim))
        generated_img = generator(noise_vec)
        disc_fake_g = discriminator(generated_img)
        loss_g = 0.5 * g_criteria(disc_fake_g, valid)  # G wants generated imgs to be labeled 1

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            loss_g.backward()
            g_optim.step()

        # collect outputs as return values
        outputs = (
            generated_img,
            imgs,
            disc_fake,
            disc_true,
            disc_fake_g,
        )
        losses = (loss_d, loss_g, loss_fake, loss_true)
        return outputs, losses

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, _, disc_fake, disc_true, _ = output
        true_positive = torch.sum(disc_true > 0.5)
        true_negative = torch.sum(disc_fake < 0.5)
        numel = torch.numel(disc_fake)

        # recall specificity
        recall = true_positive.float() / numel
        specificity = true_negative.float() / numel
        accuracy = (recall + specificity) / 2

        return {
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'loss_fake': loss[2].item(),
            'loss_true': loss[3].item(),
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
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
    with open('lsgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = LSGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

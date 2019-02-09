import random
from typing import Dict
import torch
from torch import optim
import torchvision
from .wgan_gp import WganGpDiscriminator
from .wgan_gp import WganGpGenerator
from datasets.img_popular import LSUNLoaderMaker
from base_trainer import NetworkTrainer
from base_trainer import ModelInfo
from base_trainer import TrainStage


class WGANTrainer(NetworkTrainer):
    """Trainer for Wasserstein GAN with Gradient Penalty.
    Gulrajani et al. - "Improved Training of Wasserstein GANs" (2017)
    """
    def __init__(self, config: dict):
        print('Configuration:')
        print(config)

        # parse configs
        self.batch_size = config['batch_size']
        self.input_dim = config['input_dim']
        self.total_epoch = config['epoch']
        self.display_imgs = config['display_imgs']
        self.grad_penalty_coeff = config['grad_penalty_coeff']

        # create a loader maker
        loader_maker = LSUNLoaderMaker(data_root='data_in', batch_size=self.batch_size)

        # create models
        generator = WganGpGenerator(input_dim=self.input_dim)
        discriminator = WganGpDiscriminator()
        models = {
            'WGANGP_G': ModelInfo(
                model=generator,
                input_size=(self.input_dim, ),
                metric='loss_g',
            ),
            'WGANGP_D': ModelInfo(
                model=discriminator,
                input_size=(self.input_dim, ),
                metric='loss_d',
            )
        }

        # create optimizers
        self.lr_init = config['lr_init']
        optimizers = {
            'g_optim': optim.Adam(
                generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999)),
            'd_optim': optim.Adam(
                discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        }

        # criteria = None
        super().__init__(
            models, loader_maker, None, optimizers, epoch=self.total_epoch, num_devices=4)

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
        generator = model['WGANGP_G'].model
        discriminator = model['WGANGP_D'].model

        g_optim, d_optim = optimizer['g_optim'], optimizer['d_optim']

        ones = torch.ones((self.batch_size, 1)).to(self.device)
        minus_ones = ones * -1.0

        ###############
        ### train G ###
        ###############
        noise_vec = torch.randn((self.batch_size, self.input_dim, 1, 1))
        generated_img = generator(noise_vec)
        g_loss = discriminator(generated_img)

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            g_loss.backward(ones)
            g_optim.step()

        ###############
        ### train D ###
        ###############
        noise_vec = torch.randn((self.batch_size, self.input_dim, 1, 1))

        # loss function - discriminator acts like a 'critic'
        d_loss_real = discriminator(imgs)
        d_loss_fake = discriminator(generator(noise_vec).detach())

        ### calculate the gradient penalty (GP)
        # linear-interpolated input
        alpha = random.random()
        img_interp = (imgs - generated_img) * alpha + generated_img
        img_interp = img_interp.detach().requires_grad_()  # set requires_grad=True to store the grad value
        score_img_interp = discriminator(img_interp)
        score_img_interp.backward(torch.ones(img_interp.size()))  # MUST zero_grad after calculation!
        # norm of gradients calculated per samples in batch
        # output size: [batch_size]
        grad_per_samps = img_interp.grad.view((self.batch_size, -1)).norm(dim=1)
        # get l2-norm of gradient penalty
        grad_penalty = self.grad_penalty_coeff * torch.pow(grad_per_samps - 1, 2).mean()

        d_loss = d_loss_real.mean() - d_loss_fake.mean() + grad_penalty

        if train_stage == TrainStage.TRAIN:
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

        # collect outputs as return values
        outputs = (
            generated_img,
            imgs,
            grad_penalty,
        )
        losses = (g_loss, d_loss, d_loss_real, d_loss_fake)
        return outputs, losses

    @staticmethod
    def make_performance_metric(input_, output, loss):
        return {
            'gp': output[2].item(),
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'd_loss_real': loss[2].item(),
            'd_loss_fake': loss[3].item(),
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
    with open('wgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = WGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

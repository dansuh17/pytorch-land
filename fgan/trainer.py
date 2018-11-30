import operator

import torch
import torchvision

from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, TrainStage, ModelInfo
from .fgan import FGanDiscriminator, FGanGenerator
from .divergence import GanDivergence, KLDivergence


class FGanTrainer(NetworkTrainer):
    """Trainer for f-GAN"""
    def __init__(self, config: dict):
        print('Configuration: ')
        print(config)

        latent_dim = config['latent_dim']
        g_input = (latent_dim, 1, 1)
        self.height = config['height']
        self.width = config['width']
        img_size = (1, self.height, self.width)

        self.display_imgs = config['display_imgs']
        self.batch_size = config['batch_size']
        self.epoch = config['epoch']

        self.divergence = GanDivergence

        # create models
        generator = FGanGenerator(latent_dim)
        discriminator = FGanDiscriminator(activation_func=self.divergence.output_activation)
        models = {
            'FGan_G': ModelInfo(
                model=generator, input_size=g_input, metric='loss_g', comparison=operator.lt),
            'FGan_D': ModelInfo(model=discriminator, input_size=img_size, metric='loss_d'),
        }

        # define loss functions, as defined according to the specific f-divergence metric
        criteria = {
            'd_criteria': self.divergence.d_loss_func,
            'g_criteria': self.divergence.g_loss_func,
        }

        # set data loader maker
        loader_maker = MNISTLoaderMaker(data_root='data_in', batch_size=self.batch_size)

        self.lr_init = config['lr_init']
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        # create this trainer
        super().__init__(
            models, loader_maker, criteria, optimizers, epoch=self.epoch, lr_scheduler=None)

        self.skip_g_per_epochs = -1
        self.iter_g = 1
        self.iter_d = 1

    def run_step(self, model, criteria, optimizer, input_, train_stage, *args, **kwargs):
        # required information
        imgs, _ = input_
        batch_size = imgs.size(0)

        generator = model['FGan_G'].model
        discriminator = model['FGan_D'].model

        # add noise
        # 4D noise vector : (b x latent_dim x 1 x 1)
        noise_size = (batch_size, ) + model['FGan_G'].input_size

        optimizer_g, optimizer_d = optimizer

        # must be trained at least once
        assert(self.iter_d > 0)
        assert(self.iter_g > 0)

        # generate latent noise vector - from standard normal distribution
        z = torch.randn(noise_size)

        # train discriminator
        for _ in range(self.iter_d):

            classified_fake = discriminator(generator(z))  # detach to prevent generator training
            classified_real = discriminator(imgs)

            # calculate losses
            loss_d = criteria['d_criteria'](classified_real, classified_fake)

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()
            else:
                break

        # train generator
        for _ in range(self.iter_g):
            generated = generator(z)
            classified_fake = discriminator(generated)

            loss_g = criteria['g_criteria'](classified_fake)

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
            else:
                break  # no need to iterate if not training

        # collect outputs and losses
        output = (generated, classified_fake, classified_real, z, imgs)
        loss = (loss_g, loss_d)
        return output, loss

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, score_fake, score_real, _, _ = output

        return {
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'score_fake_mean': torch.mean(score_fake).item(),
            'score_real_mean': torch.mean(score_real).item(),
        }

    def pre_epoch_finish(self, input, output, metric_manager, train_stage: TrainStage):
        """Add example images from validation step just before the end of epoch training."""
        if train_stage == TrainStage.VALIDATE:
            generated_imgs = output[0]
            real_imgs = output[-1]
            self.add_generated_image(
                generated_imgs, nrow=self.display_imgs, height=self.height,
                width=self.width, name='generated')
            self.add_generated_image(
                real_imgs, nrow=self.display_imgs, height=self.height,
                width=self.width, name='real')

    def add_generated_image(self, imgs, nrow, height, width, name: str):
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)


if __name__ == '__main__':
    # read configuration file
    import json
    with open('fgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = FGanTrainer(config)
    trainer.fit()
    trainer.cleanup()

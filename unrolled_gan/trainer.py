import operator
import copy
import torch
from typing import Dict
from torch import nn
from datasets.mixture_gaussians import MixtureOfGaussiansLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage
from .unrolled_gan import UnrolledGanDiscriminator, UnrolledGanGenerator


class UnrolledGanTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        print('Configuration:')
        print(config)

        self.input_size = 256
        self.data_dim = 2
        self.total_dataset_size = 10000
        self.batch_size = 128
        self.epoch = 100
        self.unroll = 5

        loader_maker = MixtureOfGaussiansLoaderMaker(
            total_size=self.total_dataset_size, batch_size=self.batch_size * (self.unroll + 1))

        generator = UnrolledGanGenerator(input_size=self.input_size, output_size=self.data_dim)
        discriminator = UnrolledGanDiscriminator(input_size=self.data_dim)
        models = {
            'UnrolledGAN_G': ModelInfo(
                model=generator, input_size=(self.input_size, ),
                metric='loss_g', comparison=operator.lt),
            'UnrolledGAN_D': ModelInfo(
                model=discriminator, input_size=(self.data_dim, ), metric='loss_d'),
        }

        self.lr_init = 0.002
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        criteria = {'criteria': nn.BCELoss()}

        super().__init__(
            models, loader_maker, criteria, optimizers,
            epoch=self.epoch, lr_scheduler=None)

        self.main_d = None
        self.unroll_step = 0

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 real_samples,
                 train_stage: TrainStage,
                 *args, **kwargs):
        generator = model['UnrolledGAN_G'].model
        discriminator = model['UnrolledGAN_D'].model
        criteria = criteria['criteria']
        optimizer_g, optimizer_d = optimizer

        valid = torch.ones((self.batch_size, 1)).to(self.device)
        invalid = torch.zeros((self.batch_size, 1)).to(self.device)

        # split between train batch and batch used for unrolling
        train_batch, unroll_batch = \
            real_samples[:self.batch_size], real_samples[self.batch_size:]

        # create noise vector
        z = torch.randn(self.input_size)

        ### train D
        classified_fake = discriminator(generator(z))
        classified_real = discriminator(train_batch)

        real_loss = criteria(classified_real, valid)
        fake_loss = criteria(classified_fake, invalid)
        loss_d = real_loss + fake_loss

        if train_stage == TrainStage.TRAIN:
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        ### train G
        # unroll D
        original_d = copy.deepcopy(discriminator)
        for i in range(self.unroll_step):
            samples = unroll_batch[self.batch_size * i: self.batch_size * (i + 1)]

            noise_vec = torch.randn(self.input_size)
            classified_fake = discriminator(generator(noise_vec))
            classified_real = discriminator(samples)
            real_loss = criteria(classified_real, valid)
            fake_loss = criteria(classified_fake, invalid)
            loss_d = real_loss + fake_loss

            if train_stage == TrainStage.TRAIN:
                optimizer_d.zero_grad()
                loss_d.backward(create_graph=True)
                optimizer_d.step()

        # use the D trained through unrolled steps to calculate generator loss
        generated = generator(z)
        classified_fake = discriminator(generated)
        loss_g = criteria(classified_fake, valid)  # G wants these labels 'valid'

        if train_stage == TrainStage.TRAIN:
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        # restore original discriminator after updating G
        discriminator.load(original_d)

        # TODO:
        output = (generated, classified_fake, classified_real)
        loss = (loss_g, loss_d)
        return output, loss

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, classified_fake, classified_real = output
        return {
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'classified_fake': classified_fake.item(),
            'classified_real': classified_real.item(),
        }


if __name__ == '__main__':
    trainer = UnrolledGanTrainer({})
    trainer.fit()
    trainer.cleanup()

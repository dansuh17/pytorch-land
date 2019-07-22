import operator
import os
import copy
from typing import Dict
import torch
import numpy as np
from torch import nn
from datasets.mixture_gaussians import MixtureOfGaussiansLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage
from .unrolled_gan import UnrolledGanDiscriminator, UnrolledGanGenerator


class UnrolledGanTrainer(NetworkTrainer):
    """Trainer for Unrolled GAN"""
    def __init__(self, config: dict):
        print('Configuration:')
        print(config)

        self.input_size = config['input_size']
        self.data_dim = config['data_dim']
        self.total_dataset_size = config['dataset_size']
        self.batch_size = config['batch_size']
        self.total_epoch = config['epoch']
        self.unroll = config['unroll']

        # use mixture of gaussian dataset
        loader_maker = MixtureOfGaussiansLoaderMaker(
            total_size=self.total_dataset_size,
            batch_size=self.batch_size * (self.unroll + 1))  # prepare more data for unrolling steps

        # define models
        generator = UnrolledGanGenerator(input_size=self.input_size, output_size=self.data_dim)
        discriminator = UnrolledGanDiscriminator(input_size=self.data_dim)
        models = {
            'UnrolledGAN_G': ModelInfo(
                model=generator, input_size=(self.input_size, ),
                metric='loss_g', comparison=operator.lt),
            'UnrolledGAN_D': ModelInfo(
                model=discriminator, input_size=(self.data_dim, ), metric='loss_d'),
        }

        # define optimizers
        self.lr_init = config['lr_init']
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        # define the criterion
        criteria = {'criteria': nn.BCELoss()}

        super().__init__(
            models, loader_maker, criteria, optimizers,
            epoch=self.total_epoch, lr_scheduler=None)

        # create a new output dir to save generated points
        self.generated_dir = self._create_output_dir('generated')
        self.main_d = None

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
        z = torch.randn((self.batch_size, self.input_size))

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
        original_d = copy.deepcopy(discriminator.module.state_dict())
        for i in range(self.unroll):
            samples = unroll_batch[self.batch_size * i: self.batch_size * (i + 1)]

            noise_vec = torch.randn((self.batch_size, self.input_size))
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
        discriminator.module.load_state_dict(original_d)

        # on validation stage, generate new examples for saving - useful for visualization
        generate_val = None
        if train_stage == TrainStage.VALIDATE:
            generate_val = generator(torch.randn(500, self.input_size))

        output = (generated, classified_fake, classified_real, generate_val)
        loss = (loss_g, loss_d)
        return output, loss

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, classified_fake, classified_real, _ = output
        batch_size = classified_fake.size(0)
        true_positive = torch.sum(classified_real > 0.5)
        true_negative = torch.sum(classified_fake < 0.5)

        recall = true_positive.float() / batch_size
        specificity = true_negative.float() / batch_size

        return {
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
        }

    def pre_epoch_finish(self, input_, output, metric_manager, train_stage: TrainStage):
        # save generated points
        if train_stage == TrainStage.VALIDATE:
            generated = output[3].detach().numpy()
            fname = 'generated_e{}'.format(self.epoch)
            np.save(os.path.join(self.generated_dir, fname), generated)


if __name__ == '__main__':
    # read configuration file
    import json
    with open('unrolled_gan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = UnrolledGanTrainer(config)
    trainer.fit()
    trainer.cleanup()

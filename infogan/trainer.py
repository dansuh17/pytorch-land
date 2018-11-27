import operator
from typing import Dict

import torch
from torch import nn
import torchvision
import numpy as np

from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, TrainStage, ModelInfo
from .infogan import InfoGanMnistGenerator, InfoGanMnistDiscriminator


class InfoGanTrainer(NetworkTrainer):
    """Trainer for InfoGAN"""
    def __init__(self, config: dict):
        print('Configuration: ')
        print(config)

        # info about latent vector / code dimensions
        self.noise_size = config['noise_size']
        self.cont_code_size = config['continuous_code_size']
        self.disc_code_size = config['discrete_code_size']
        self.code_size = self.cont_code_size + self.disc_code_size
        self.input_size = self.noise_size + self.code_size

        self.height = config['height']
        self.width = config['width']
        img_size = (1, self.height, self.width)
        g_input = (self.input_size, 1, 1)

        self.display_imgs = config['display_imgs']
        self.batch_size = config['batch_size']
        self.epoch = config['epoch']
        self.lambda_cont = config['lambda_cont']  # lambda term for continuous var
        self.lambda_disc = config['lambda_disc']  # lambda term for discrete var

        # create models
        generator = InfoGanMnistGenerator()
        discriminator = InfoGanMnistDiscriminator(img_size, self.noise_size, self.code_size)
        models = {
            'InfoGAN_G': ModelInfo(
                model=generator, input_size=g_input, metric='loss_g', comparison=operator.lt),
            'InfoGAN_D': ModelInfo(model=discriminator, input_size=img_size, metric='loss_d'),
        }

        # set data loader maker
        loader_maker = MNISTLoaderMaker(
            data_root='data_in', batch_size=self.batch_size, naive_normalization=True)

        # create criteria
        d_criterion = nn.BCELoss()  # binary cross entropy loss
        disc_code_criterion = nn.CrossEntropyLoss()  # discrete code loss
        cont_code_criterion = nn.MSELoss()  # continuous code loss
        criteria = (d_criterion, disc_code_criterion, cont_code_criterion)

        # create optimizers
        lr_init_g = config['lr_init_g']
        lr_init_d = config['lr_init_d']
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_init_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_init_d, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        # learning rate schedulers
        # TODO: validate the effects of schedulers
        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', verbose=True, factor=0.9, patience=10)
        lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d, mode='min', verbose=True, factor=0.9, patience=10)

        # create this trainer
        super().__init__(
            models, loader_maker, criteria, optimizers, epoch=self.epoch, lr_scheduler=None)

        self.test_vector = self.generate_test_vector()

        # TODO: move to config?
        self.iter_g = 1
        self.iter_d = 1

    def create_noise_vector(self, batch_size: int, noise_size: int):
        """
        Creates the random noise vector for input.

        Args:
            batch_size (int): batch size
            noise_size (int): noise vector length

        Returns:
            noise vector 'z' with size (batch_size, noise_size)
        """
        noise_dim = (batch_size, noise_size)
        return torch.randn(noise_dim).to(self.device)

    def create_discrete_latent_code(
            self, batch_size: int, code_size: int, pick_idx: int=None, random_val=False):
        """
        Creates discrete latent code vector.

        Args:
            batch_size (int): batch size
            code_size (int): discrete latent code length

        Returns:
            discrete latent code with size (batch_size, code_size)
        """
        if random_val:
            single_prob = 1 / code_size
            prob_array = [single_prob] * code_size  # probability follows uniform distribution
        else:
            assert pick_idx is not None and pick_idx < code_size
            prob_array = [0.0] * code_size
            prob_array[pick_idx] = 1.0  # mark one index
        onehot_vec = np.random.multinomial(n=1, pvals=prob_array, size=batch_size)
        return torch.from_numpy(onehot_vec).float().to(self.device)

    def create_continuous_latent_code(self, batch_size: int, code_size=1):
        """
        Creates continuous latent code vector.

        Args:
            batch_size (int): batch size
            code_size (int): continuous latent code length

        Returns:
            continuos latent code vector with size (batch_size, code_size)
        """
        random_uniform_vec = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, code_size))
        return torch.from_numpy(random_uniform_vec).float().to(self.device)

    def create_input(self, batch_size, noise_size, disc_code_size, cont_code_size):
        """
        Creates input - this is a concatenation of :
        [noise vector, discrete latent code, continuous latent code]

        Args:
            batch_size (int): batch size
            noise_size (int): noise vector length
            disc_code_size (int): discrete code vector length
            cont_code_size (int): continuous code vector length

        Returns:
            input vector of size (batch_size, (noise_size + disc_code_size + cont_code_size), 1)
        """
        z = self.create_noise_vector(batch_size, noise_size=noise_size)
        c_disc = self.create_discrete_latent_code(batch_size, code_size=disc_code_size, random_val=True)
        c_cont = self.create_continuous_latent_code(batch_size, code_size=cont_code_size)
        return torch.cat((z, c_disc, c_cont), dim=1).to(self.device)

    def parse_latent(self, latent_vector, noise_size: int,
                     disc_code_size: int, cont_code_size: int):
        """
        Splits the latent vector into [noise vector, discrete latent code, continuous latent code]

        Args:
            latent_vector (torch.Tensor):
            noise_size (int): noise vector size
            disc_code_size (int): discrete code vector size
            cont_code_size (int): continuous code vector size

        Returns:
            (noise_vector, discrete_latent_code, continuous_latent_code)
        """
        return torch.split(
            latent_vector,
            split_size_or_sections=[noise_size, disc_code_size, cont_code_size],
            dim=1)

    def parse_disc_output_code(self, disc_output_code):
        """
        Splits the output of discriminator's code vector into :
        [discrete_latent_code_logits, reconstructed_continuous_code]

        Args:
            disc_output_code (torch.Tensor): discriminator output

        Returns:
            (discrete_logits, reconstructed_cont_code)
        """
        return torch.split(
            disc_output_code, [self.disc_code_size, self.cont_code_size], dim=1)

    def run_step(self, model: Dict[str, ModelInfo],
                 criteria, optimizer, input_, train_stage, *args, **kwargs):
        # required information
        imgs, _ = input_
        batch_size = imgs.size(0)

        # create target values
        valid = torch.ones((batch_size, 1)).to(self.device)  # mark valid
        invalid = torch.zeros((batch_size, 1)).to(self.device)  # mark invalid

        # prepare materials for training
        generator = model['InfoGAN_G'].model
        discriminator = model['InfoGAN_D'].model
        optimizer_g, optimizer_d = optimizer
        d_crit, disc_code_crit, cont_code_crit = criteria

        # must be trained at least once
        assert(self.iter_d > 0)
        assert(self.iter_g > 0)

        # generate input
        latent_vec = self.create_input(
            batch_size,
            noise_size=self.noise_size,
            disc_code_size=self.disc_code_size,
            cont_code_size=self.cont_code_size)
        # separate representation for loss calculation
        noise_vector, disc_code_in, cont_code_in = self.parse_latent(
            latent_vec, self.noise_size, self.disc_code_size, self.cont_code_size)

        # train discriminator
        for _ in range(self.iter_d):
            # detach to prevent generator training
            classified_fake, code_prob = discriminator(generator(latent_vec))
            classified_real, _ = discriminator(imgs)

            # parse latent code outputs
            disc_code_out, cont_code_out = self.parse_disc_output_code(code_prob)

            # calculate information loss
            _, target_codes = disc_code_in.max(dim=1)  # max over 1st dimension
            disc_loss = disc_code_crit(disc_code_out, target_codes)  # cross entropy
            cont_loss = cont_code_crit(cont_code_out, cont_code_in)  # mean squared error
            info_loss = self.lambda_disc * disc_loss + self.lambda_cont * cont_loss

            # calculate losses
            fake_loss = d_crit(classified_fake, invalid)
            real_loss = d_crit(classified_real, valid)
            loss_d = real_loss + fake_loss + info_loss

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()
            else:
                break

        # train generator + information loss backprop
        for _ in range(self.iter_g):
            # generated images
            generated = generator(latent_vec)

            # generator wants to make generated images 'valid'
            loss_g = d_crit(classified_fake, valid) + info_loss

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

        # collect outputs and losses
        output = (
            generated,
            classified_fake,
            classified_real,
            noise_vector,
            disc_code_in,
            cont_code_in,
            imgs
        )
        loss = (loss_g, loss_d, fake_loss, real_loss, info_loss, disc_loss, cont_loss)
        return output, loss

    @property
    def standard_metric(self):
        return 'g_loss_with_info', 'd_loss'  # must have each for each model

    @staticmethod
    def make_performance_metric(input_, output, loss):
        classified_fake, classified_real = output[1], output[2]
        true_negative = torch.sum(classified_fake < 0.5)
        true_positive = torch.sum(classified_real > 0.5)
        numel = torch.numel(classified_fake)

        # calculate various statistics for discriminator's performance
        specificity = true_negative.float() / numel
        recall = true_positive.float() / numel
        accuracy = (specificity + recall) / 2.0
        return {
            'g_loss': loss[0].item(),
            'g_loss_with_info': loss[0].item() + loss[4].item(),
            'd_loss': loss[1].item(),
            'd_loss_fake': loss[2].item(),
            'd_loss_real': loss[3].item(),
            'info_loss': loss[4].item(),
            'disc_loss': loss[5].item(),
            'cont_loss': loss[6].item(),
            'd_accuracy': accuracy.item(),
            'd_specificity': specificity.item(),
            'd_recall': recall.item(),
        }

    def generate_test_vector(self):
        # generate single noise input
        z = self.create_noise_vector(1, noise_size=self.noise_size)
        c_cont = self.create_continuous_latent_code(1, code_size=self.cont_code_size)

        # change only the discrete variable
        vectors = []
        for i in range(self.disc_code_size):
            c_disc = self.create_discrete_latent_code(
                1, code_size=self.disc_code_size, pick_idx=i, random_val=False)
            test_vector_batch = torch.cat((z, c_disc, c_cont), dim=1)
            vectors.append(test_vector_batch)
        return torch.cat(tuple(vectors), dim=0).to(self.device)

    def pre_epoch_finish(self, input, output, metric_manager, train_stage: TrainStage):
        """Add example images from validation step just before the end of epoch training."""
        generator = self.models['InfoGAN_G'].model
        # requires the tensor to be in CPU to convert to numpy
        test_gen_imgs = generator(self.generate_test_vector()).detach().cpu()

        if train_stage == TrainStage.VALIDATE:
            generated_imgs, real_imgs = output[0], output[-1]
            self.add_generated_image(
                generated_imgs, nrow=self.display_imgs, name='generated')
            self.add_generated_image(
                real_imgs, nrow=self.display_imgs, name='real')
            # add test images for discrete codes
            self.add_generated_image(
                test_gen_imgs, nrow=self.disc_code_size, name='discrete_test_imgs')

    def add_generated_image(self, imgs, nrow, name: str):
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)


if __name__ == '__main__':
    # read configuration file
    import json
    with open('infogan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = InfoGanTrainer(config)
    trainer.fit()
    trainer.cleanup()

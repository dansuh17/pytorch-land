import torch
from torch import nn
import torchvision

from datasets.img_popular import LSUNLoaderMaker
from base_trainer import NetworkTrainer, TrainStage
from .dcgan import DCGANDiscriminator, DCGANGenerator


class DCGANTrainer(NetworkTrainer):
    """Trainer for DCGAN"""
    def __init__(self, config: dict):
        print('Configuration: ')
        print(config)
        latent_dim = config['latent_dim']
        g_input = (latent_dim, )
        self.height = config['height']
        self.width = config['weight']
        img_size = (1, self.height, self.width)
        inputs = (g_input, img_size)
        self.display_imgs = config['display_imgs']
        self.batch_size = config['batch_size']
        self.lr_init = config['lr_init']
        self.epoch = config['epoch']

        # create models
        generator = DCGANGenerator(input_dim=latent_dim)
        discriminator = DCGANDiscriminator()
        models = (generator, discriminator)

        # set data loader maker
        loader_maker = LSUNLoaderMaker(data_root='data_in', batch_size=self.batch_size)
        criterion = nn.BCELoss()  # binary cross entropy loss

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=self.lr_init, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        # TODO: validate the effects of schedulers
        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', verbose=True, factor=0.9, patience=10)
        lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d, mode='min', verbose=True, factor=0.9, patience=10)

        # create this trainer
        super().__init__(
            models, loader_maker, criterion, optimizers,
            epoch=self.epoch, input_size=inputs, lr_scheduler=None)

        self.skip_g_per_epochs = -1
        self.iter_g = 1
        self.iter_d = 1

    def run_step(self, model, criteria, optimizer, input_, train_stage):
        # required information
        imgs, _ = input_
        batch_size = imgs.size(0)

        # add noise
        latent_dim = self.input_size[0]
        # TODO: try label switching - valid is marked 0, invalid is marked 1
        valid = torch.ones((batch_size, 1)).to(self.device)  # mark valid
        invalid = torch.zeros((batch_size, 1)).to(self.device)  # mark invalid

        generator, discriminator = model
        optimizer_g, optimizer_d = optimizer

        # must be trained at least once
        assert(self.iter_d > 0)
        assert(self.iter_g > 0)

        # train discriminator
        for _ in range(self.iter_d):
            z = torch.randn((batch_size, ) + latent_dim)

            classified_fake = discriminator(generator(z).detach())  # detach to prevent generator training
            classified_real = discriminator(imgs)

            # calculate losses
            fake_loss = criteria(classified_fake, invalid)
            real_loss = criteria(classified_real, valid)
            loss_d = real_loss + fake_loss

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
            else:
                break

        # train generator
        for _ in range(self.iter_g):
            # generate latent noise vector - from standard normal distribution
            z = torch.randn((batch_size, ) + latent_dim).to(self.device)

            generated = generator(z)
            classified_fake = discriminator(generated)

            loss_g = criteria(classified_fake, valid)  # generator wants to make generated images 'valid'

            # update parameters if training
            if train_stage == TrainStage.TRAIN and self.epoch > self.skip_g_per_epochs:
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
            else:
                break  # no need to iterate if not training

        # collect outputs and losses
        output = (generated, classified_fake, classified_real, z, imgs)
        loss = (loss_g, loss_d, fake_loss, real_loss)

        if train_stage == TrainStage.TRAIN:
            self.train_step += 1

        return output, loss

    @property
    def standard_metric(self):
        return 'g_loss', 'd_loss'  # must have each for each model

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, classified_fake, classified_real, _, _ = output
        true_negative = torch.sum(classified_fake < 0.5)
        true_positive = torch.sum(classified_real > 0.5)
        numel = torch.numel(classified_fake)

        # calculate various statistics for discriminator's performance
        specificity = true_negative.float() / numel
        recall = true_positive.float() / numel
        accuracy = (specificity + recall) / 2.0
        return {
            'g_loss': loss[0].item(),
            'd_loss': loss[1].item(),
            'd_loss_fake': loss[2].item(),
            'd_loss_real': loss[3].item(),
            'd_accuracy': accuracy.item(),
            'd_specificity': specificity.item(),
            'd_recall': recall.item(),
        }

    def pre_epoch_finish(self, input, output, metric_manager, train_stage: TrainStage):
        """Add example images from validation step just before the end of epoch training."""
        if train_stage == TrainStage.VALIDATE:
            generated_imgs, _, _, _, real_imgs = output
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
    with open('dcgan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = DCGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

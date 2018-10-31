import torch
from torch import nn
import torchvision

from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, TrainStage
from .gan import Generator, Discriminator


class GanTrainer(NetworkTrainer):
    def __init__(self):
        g_input = (100, )
        self.height = 28
        self.width = 28
        img_size = (1, self.height, self.width)
        inputs = (g_input, img_size)
        generator = Generator(input_dim=100, img_size=img_size)
        discriminator = Discriminator(img_size=img_size)
        models = (generator, discriminator)

        loader_maker = MNISTLoaderMaker(data_root='data_in', batch_size=128)
        criterion = nn.BCELoss()  # binary cross entropy loss

        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)

        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', verbose=True, factor=0.2, patience=7)
        lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d, mode='min', verbose=True, factor=0.2, patience=7)
        super().__init__(
            models, loader_maker, criterion, optimizers,
            epoch=100, input_size=inputs, lr_scheduler=(lr_scheduler_g, lr_scheduler_d))

        self.train_g_after_epoch = 1
        self.iter_g = 1
        self.iter_d = 2

    def run_step(self, model, criteria, optimizer, input_, train_stage):
        # required information
        imgs = input_[0]
        batch_size = imgs.size()[0]
        latent_dim = self.input_size[0]
        ones = torch.ones((batch_size, 1))  # mark valid
        zeros = torch.zeros((batch_size, 1))  # mark invalid

        generator, discriminator = model
        optimizer_g, optimizer_d = optimizer

        # must be trained at least once
        assert(self.iter_d > 0)
        assert(self.iter_g > 0)

        # train generator
        for _ in range(self.iter_g):
            # generate latent noise vector
            noise = torch.randn((batch_size, ) + latent_dim)

            generated = generator(noise)
            classified_fake = discriminator(generated)

            loss_g = criteria(classified_fake, ones)  # generator wants to make generated images 'valid'

            # update parameters if training
            if train_stage == TrainStage.TRAIN and self.epoch > self.train_g_after_epoch:
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
            else:
                break  # no need to iterate if not training

        # reuse generated image without affecting the graph for the generator
        generated = generated.detach()

        for _ in range(self.iter_d):
            # train discriminator
            classified_fake = discriminator(generated)
            classified_real = discriminator(imgs)

            fake_loss = criteria(classified_fake, zeros)
            real_loss = criteria(classified_real, ones)
            loss_d = (real_loss + fake_loss) / 2

            # update parameters if training
            if train_stage == TrainStage.TRAIN:
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
            else:
                break

        # collect outputs and losses
        output = (generated, classified_fake, classified_real, noise)
        loss = (loss_g, loss_d)

        if train_stage == TrainStage.TRAIN:
            self.train_step += 1

        return output, loss

    @property
    def standard_metric(self):
        return 'g_loss', 'd_loss'  # must have each for each model

    @staticmethod
    def make_performance_metric(input_, output, loss):
        _, classified_fake, classified_real, _ = output
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
            'd_accuracy': accuracy.item(),
            'd_specificity': specificity.item(),
            'd_recall': recall.item(),
        }

    def pre_epoch_finish(self, input, output, metric_manager, train_stage: TrainStage):
        if train_stage == TrainStage.VALIDATE:
            generated_imgs, _, _, _ = output
            self.add_generated_image(
                generated_imgs, nrow=4, height=self.height, width=self.width, name='generated')

    def add_generated_image(self, imgs, nrow, height, width, name: str):
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)


if __name__ == '__main__':
    trainer = GanTrainer()
    trainer.fit()
    trainer.cleanup()

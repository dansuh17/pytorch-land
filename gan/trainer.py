import torch
from torch import nn
from datasets.img_popular import MNISTLoaderMaker
from base_trainer import NetworkTrainer, TrainStage
from .gan import Generator, Discriminator


class GanTrainer(NetworkTrainer):
    def __init__(self):
        lr_init = 0.0002
        g_input = (100, )
        img_size = (1, 28, 28)
        inputs = (g_input, img_size)
        generator = Generator(input_dim=100, img_size=img_size)
        discriminator = Discriminator(img_size=img_size)
        models = (generator, discriminator)
        loader_maker = MNISTLoaderMaker(data_root='data_in', batch_size=128)
        criterion = nn.BCELoss()  # binary cross entropy loss
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_init, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_init, betas=(0.5, 0.999))
        optimizers = (optimizer_g, optimizer_d)
        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='min', verbose=True, factor=0.2, patience=7)
        lr_scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d, mode='min', verbose=True, factor=0.2, patience=7)
        super().__init__(
            models, loader_maker, criterion, optimizers,
            epoch=100, input_size=inputs, lr_scheduler=(lr_scheduler_g, lr_scheduler_d))

    def run_step(self, model, criteria, input_, train_stage):
        # required information
        batch_size = input_[0]
        latent_dim = self.input_size[0]
        ones = torch.ones((batch_size, ))
        zeros = torch.zeros((batch_size, ))

        generator, discriminator = model
        optimizer_g, optimizer_d = self.optimizer

        # generate latent noise vector
        noise = torch.randn((batch_size, ) + latent_dim)

        # train generator
        generated = generator(noise)
        classified_fake = discriminator(generated)

        loss_g = criteria(classified_fake, ones)

        if train_stage == TrainStage.TRAIN:
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        # train discriminator
        classified_fake = classified_fake.detach()  # prevent training generator
        classified_real = discriminator(input_)
        fake_loss = criteria(classified_fake, zeros)
        real_loss = criteria(classified_real, ones)
        loss_d = (real_loss + fake_loss) / 2

        if train_stage == TrainStage.TRAIN:
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        output = (generated, classified_fake, classified_real, noise)
        loss = (loss_g, loss_d)

        if train_stage == TrainStage.TRAIN:
            self.train_step += 1

        return output, loss

    @staticmethod
    def make_performance_metric(input_, output, loss):
        return {'g_loss': loss[0], 'd_loss': loss[1]}


if __name__ == '__main__':
    trainer = GanTrainer()
    trainer.fit()
    trainer.cleanup()

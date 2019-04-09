import operator
from typing import Dict
import torch
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
import torchvision
from cyclegan.cyclegan import CycleGanDiscriminator, CycleGanGenerator
from datasets.img_transfer import Monet2PhotoLoaderMaker
from base_trainer import NetworkTrainer, ModelInfo, TrainStage


class CycleGANTrainer(NetworkTrainer):
    """
    Trainer for CycleGAN
    """
    def __init__(self, config: dict):
        print('Configuration: ')
        print(config)

        self.batch_size = config['batch_size']
        self.data_root_dir = config['data_root']
        self.total_epoch = config['epoch']
        self.lr = config['lr']
        self.use_id_loss = config['use_id_loss']
        self.cycle_loss_lambda = config['cycle_loss_lambda']  # typically 10
        if self.use_id_loss:
            self.id_loss_lambda = config['id_loss_lambda']  # typically 5
        self.num_devices = config['num_device']
        self.display_imgs = 10  # display 10 sample images per epoch

        loader_maker = Monet2PhotoLoaderMaker(
            self.batch_size, self.data_root_dir, downsize_half=True, num_workers=4)
        img_size = loader_maker.img_size  # input size

        # create models
        g = CycleGanGenerator()  # monet -> photo generator
        f = CycleGanGenerator()  # photo -> monet generator
        d_x = CycleGanDiscriminator()  # monet discriminator
        d_y = CycleGanDiscriminator()  # photo discriminator

        self.d_out_size = CycleGanDiscriminator.output_size

        models = {
            'CycleGAN_G': ModelInfo(
                model=g,
                input_size=img_size,
                metric='loss_g',  # used for saving the 'best' model
                comparison=operator.lt,
            ),
            'CycleGAN_F': ModelInfo(
                model=f,
                input_size=img_size,
                metric='loss_f',
            ),
            'CycleGAN_Dx': ModelInfo(
                model=d_x,
                input_size=img_size,
                metric='loss_d_x',
            ),
            'CycleGAN_Dy': ModelInfo(
                model=d_y,
                input_size=img_size,
                metric='loss_d_y',
            ),
        }

        criteria = {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss(),
        }

        optimizers = {
            'optimizer_g': optim.Adam(g.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_f': optim.Adam(f.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_d_x': optim.Adam(d_x.parameters(), self.lr, betas=(0.5, 0.999)),
            'optimizer_d_y': optim.Adam(d_y.parameters(), self.lr, betas=(0.5, 0.999)),
        }

        super().__init__(
            models, loader_maker, criteria, optimizers,
            epoch=self.total_epoch, num_devices=self.num_devices, lr_scheduler=None)

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria: Dict[str, _Loss],
                 optimizer: Dict[str, Optimizer],
                 input_: torch.Tensor,
                 train_stage: TrainStage,
                 *args, **kwargs):
        # input size: (batch_num, 2, channel, height, width)
        monet_real, photo_real = input_[:, 0, :, :], input_[:, 1, :, :]

        # (b, 1, 6, 6)
        ones = torch.ones((self.batch_size, ) + self.d_out_size).to(self.device)
        zeros = torch.zeros((self.batch_size, ) + self.d_out_size).to(self.device)

        # parse models
        G = model['CycleGAN_G'].model
        F = model['CycleGAN_F'].model
        Dx = model['CycleGAN_Dx'].model
        Dy = model['CycleGAN_Dy'].model

        mse_loss = criteria['mse']
        l1_loss = criteria['l1']

        g_optim = optimizer['optimizer_g']
        f_optim = optimizer['optimizer_f']
        d_x_optim = optimizer['optimizer_d_x']
        d_y_optim = optimizer['optimizer_d_y']

        ### Generate images
        # monet -> photo
        gen_photo = G(monet_real)

        # photo -> monet
        gen_monet = F(photo_real)

        ### Train generators
        # G: monet->photo
        photo_gen_score = Dy(gen_photo)
        g_loss = mse_loss(photo_gen_score, ones)

        # F: photo->monet
        monet_gen_score = Dx(gen_monet)
        f_loss = mse_loss(monet_gen_score, ones)

        # Cycle-consistency loss
        # monet -> photo -> monet
        monet_reconstructed = F(gen_photo)
        cycle_fg = l1_loss(monet_real, monet_reconstructed)

        # photo -> monet -> photo
        photo_reconstructed = G(gen_monet)
        cycle_gf = l1_loss(photo_real, photo_reconstructed)

        cycle_loss = self.cycle_loss_lambda * (cycle_gf + cycle_fg)

        # add all generator losses
        generator_loss = g_loss + f_loss + cycle_loss

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            f_optim.zero_grad()
            generator_loss.backward()
            g_optim.step()
            f_optim.step()

        ### Train Discriminators
        # Dy (photo discriminator)
        photo_gen_score = Dy(gen_photo.detach())
        photo_real_score = Dy(photo_real)
        d_y_loss_real = mse_loss(photo_real_score, ones)
        d_y_loss_fake = mse_loss(photo_gen_score, zeros)
        d_y_loss = d_y_loss_real + d_y_loss_fake

        # Dx (monet discriminator) ###
        monet_gen_score = Dx(gen_monet.detach())
        monet_real_score = Dx(monet_real)
        d_x_loss_real = mse_loss(monet_real_score, ones)
        d_x_loss_fake = mse_loss(monet_gen_score, zeros)
        d_x_loss = d_x_loss_real + d_x_loss_fake

        if train_stage == TrainStage.TRAIN:
            d_x_optim.zero_grad()
            d_x_loss.backward()
            d_x_optim.step()

            d_y_optim.zero_grad()
            d_y_loss.backward()
            d_y_optim.step()

        ### Identity Mapping Loss
        if self.use_id_loss:
            gen_photo = G(monet_real)
            id_loss_photo = l1_loss(gen_photo, photo_real)

            gen_monet = F(photo_real)
            id_loss_monet = l1_loss(gen_monet, monet_real)

            id_loss = self.id_loss_lambda * (id_loss_photo + id_loss_monet)

            if train_stage == TrainStage.TRAIN:
                g_optim.zero_grad()
                f_optim.zero_grad()
                cycle_loss.backward()
                g_optim.step()
                f_optim.step()

        outputs = (monet_real, photo_real, gen_monet, gen_photo)
        losses = [g_loss, f_loss, d_x_loss, d_y_loss, cycle_loss, cycle_fg, cycle_gf]
        if self.use_id_loss:
            losses.append(id_loss)

        return outputs, tuple(losses)

    @staticmethod
    def make_performance_metric(input_, output, loss) -> dict:
        metrics = {
            'g_loss': loss[0].item(),
            'f_loss': loss[1].item(),
            'd_x_loss': loss[2].item(),
            'd_y_loss': loss[3].item(),
            'cycle_loss': loss[4].item(),
            'cycle_fg': loss[5].item(),
            'cycle_gf': loss[6].item(),
        }
        if len(loss) == 8:
            metrics['id_loss'] = loss[7].item()
        return metrics

    def pre_epoch_finish(self, input_, output, metric_manager, train_stage: TrainStage):
        # save images at the end of validation step
        if train_stage == TrainStage.VALIDATE:
            monet_imgs, photo_imgs, gen_monet, gen_photo = output
            self.log_images(monet_imgs, nrow=self.display_imgs, name='monet_real')
            self.log_images(photo_imgs, nrow=self.display_imgs, name='photo_real')
            self.log_images(gen_monet, nrow=self.display_imgs, name='monet_gen')
            self.log_images(gen_photo, nrow=self.display_imgs, name='photo_gen')

    def log_images(self, imgs, nrow: int, name: str):
        """
        Save images to the summary writer.

        Args:
            imgs (torch.Tensor): image tensors
            nrow (int): number of images to save
            name (str): name for the images
        """
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image(f'{self.epoch}/{name}', grid, self.global_step)


if __name__ == '__main__':
    import json
    with open('cyclegan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = CycleGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

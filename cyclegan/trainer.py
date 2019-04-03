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
        self.num_devices = 4
        self.display_imgs = 10

        loader_maker = Monet2PhotoLoaderMaker(
            self.batch_size, self.data_root_dir, downsize_half=True)
        img_size = (3, 128, 128)

        g = CycleGanGenerator()
        f = CycleGanGenerator()
        d_x = CycleGanDiscriminator()
        d_y = CycleGanDiscriminator()

        models = {
            'CycleGAN_G': ModelInfo(
                model=g,
                input_size=img_size,
                metric='loss_g',
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
        monet_imgs, photo_imgs = input_[:, 0, :, :], input_[:, 1, :, :]

        ones = torch.ones((self.batch_size, 6, 6)).to(self.device)
        zeros = torch.zeros((self.batch_size, 6, 6)).to(self.device)

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

        ### Train G ###
        gen_photos = G(monet_imgs)
        photo_gen_score = Dy(gen_photos)
        g_loss = mse_loss(photo_gen_score, ones)

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        ### Train Dy (photo discriminator) ###
        gen_photos = G(monet_imgs)
        photo_gen_score = Dy(gen_photos.detach())
        photo_real_score = Dy(photo_imgs)
        d_y_loss_real = mse_loss(photo_real_score, ones)
        d_y_loss_fake = mse_loss(photo_gen_score, zeros)
        d_y_loss = d_y_loss_real + d_y_loss_fake

        if train_stage == TrainStage.TRAIN:
            d_y_optim.zero_grad()
            d_y_loss.backward()
            d_y_optim.step()

        ### Train F ###
        gen_monet = F(photo_imgs)
        monet_gen_score = Dx(gen_monet)
        f_loss = mse_loss(monet_gen_score, ones)

        if train_stage == TrainStage.TRAIN:
            f_optim.zero_grad()
            f_loss.backward()
            f_optim.step()

        ### Train Dx (monet discriminator) ###
        gen_monet = F(photo_imgs)
        monet_gen_score = Dx(gen_monet)
        monet_real_score = Dx(monet_imgs)
        d_x_loss_real = mse_loss(monet_real_score, ones)
        d_x_loss_fake = mse_loss(monet_gen_score, zeros)
        d_x_loss = d_x_loss_real + d_x_loss_fake

        if train_stage == TrainStage.TRAIN:
            d_x_optim.zero_grad()
            d_x_loss.backward()
            d_x_optim.step()

        ### Cycle-Consistency Loss
        # monet -> photo -> monet
        gen_photo = G(monet_imgs)
        monet_reconstructed = F(gen_photo)
        # cycle consistency loss for F(G(x))
        cycle_fg = l1_loss(monet_imgs, monet_reconstructed)

        # photo -> monet -> photo
        gen_monet = F(photo_imgs)
        photo_reconstructed = G(gen_monet)
        cycle_gf = l1_loss(photo_imgs, photo_reconstructed)

        cycle_loss = self.cycle_loss_lambda * (cycle_fg + cycle_gf)

        if train_stage == TrainStage.TRAIN:
            g_optim.zero_grad()
            f_optim.zero_grad()
            cycle_loss.backward()
            g_optim.step()
            f_optim.step()

        ### Identity Mapping Loss
        if self.use_id_loss:
            gen_photo = G(monet_imgs)
            id_loss_photo = l1_loss(gen_photo, photo_imgs)

            gen_monet = F(photo_imgs)
            id_loss_monet = l1_loss(gen_monet, monet_imgs)

            id_loss = self.id_loss_lambda * (id_loss_photo + id_loss_monet)

            if train_stage == TrainStage.TRAIN:
                g_optim.zero_grad()
                f_optim.zero_grad()
                cycle_loss.backward()
                g_optim.step()
                f_optim.step()

        outputs = (monet_imgs, photo_imgs, gen_monet, gen_photo)
        losses = [g_loss, f_loss, d_x_loss, d_y_loss, cycle_loss]
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
        }
        if len(loss) == 6:
            metrics['id_loss'] = loss[5].item()
        return metrics

    def pre_epoch_finish(self, input_, output, metric_manager, train_stage: TrainStage):
        # save images at the end of validation step
        if train_stage == TrainStage.VALIDATE:
            monet_imgs, photo_imgs, gen_monet, gen_photo = output
            self.log_images(monet_imgs, nrow=self.display_imgs, name='monet_real')
            self.log_images(photo_imgs, nrow=self.display_imgs, name='photo_real')
            self.log_images(gen_monet, nrow=self.display_imgs, name='monet_gen')
            self.log_images(gen_photo, nrow=self.display_imgs, name='photo_gen')

    def log_images(self, imgs, nrow, name: str):
        """
        Save images to the summary writer.

        Args:
            imgs (torch.Tensor): image tensors
            nrow (int): number of images to save
            name (str): name for the images
        """
        grid = torchvision.utils.make_grid(imgs[:nrow, :], nrow=nrow, normalize=True)
        self.writer.add_image(f'{self.epoch}/{name}', grid, self.train_step)


if __name__ == '__main__':
    import json
    with open('cyclegan/config.json', 'r') as configf:
        config = json.loads(configf.read())

    trainer = CycleGANTrainer(config)
    trainer.fit()
    trainer.cleanup()

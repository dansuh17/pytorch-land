import os
from typing import Dict
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from .schmidt_sda import SchmidtSDA
from .dae_unet import DansuhDenoisingCNN
from base_trainer import NetworkTrainer, TrainStage, ModelInfo
from utils.spectrogram import denormalize_db_spectrogram


class DansuhNetTrainer(NetworkTrainer):
    """Trainer for training dansuh denoising net."""
    def __init__(self, config: dict):
        # prepare inputs for trainer
        self.input_channel = config['input_channel']
        self.input_height = config['input_width']
        self.input_width = config['input_height']
        self.lr_init = config['lr_init']
        self.total_epoch = config['epoch']
        batch_size = config['batch_size']
        input_data_dir = config['data_dir']

        model = DansuhDenoisingCNN()
        models = {
            'DansuhNet': ModelInfo(
                model=model,
                input_size=(1, self.input_height, self.input_width),
                metric='loss',
            )
        }

        dataloader_maker = VCTKLoaderMaker(
            input_data_dir, batch_size, use_channel=True, use_db_spec=True)

        # mean squared error loss
        criterion = {
            'mse_loss': nn.MSELoss(reduction='elementwise_mean')
        }

        optimizer = {
            'adam': optim.Adam(
                params=model.parameters(), lr=self.lr_init),
        }

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer['adam'], mode='min', verbose=True, factor=0.2, patience=7)

        # initialize the trainer
        super().__init__(
            models, dataloader_maker, criterion, optimizer,
            epoch=self.total_epoch, lr_scheduler=lr_scheduler)

    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        dansuh_net = model['DansuhNet'].model

        clean_img, noisy_img = input_
        output = dansuh_net(noisy_img)
        loss = criteria['mse_loss'](output, clean_img)

        opt = optimizer['adam']
        # update the model if training
        if train_stage == TrainStage.TRAIN:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return output, loss

    @staticmethod
    def input_transform(data):
        clean_img = data[0].float()  # these have been read as double tensor
        noisy_img = data[1].float()
        return clean_img, noisy_img

    def pre_epoch_finish(self, input, output, metric_manager, train_stage):
        if train_stage == TrainStage.VALIDATE:
            # expand variables just to read easily
            clean_imgs, noisy_imgs = input
            nrow = 6
            self.add_image(clean_imgs, nrow, height=self.input_height, width=self.input_width, name='clean')
            self.add_image(noisy_imgs, nrow, height=self.input_height, width=self.input_width, name='noisy')
            self.add_image(output, nrow, height=self.input_height, width=self.input_width, name='denoised')

    def add_image(self, img, nrow, height, width, name: str):
        spec = self.make_grid_from_mel(img[:nrow, :].view(-1, 1, height, width))
        grid = torchvision.utils.make_grid(spec, nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)

    @staticmethod
    def make_grid_from_mel(imgs, sr=16000, n_fft=256, n_mels=40):  # TODO: acquire these from constants
        """
        Make image grid from a number of normalized db-scale mel-spectrograms.

        Args:
            imgs (list[torch.FloatTensor]): list of tensors representing images
            sr (int): sample rate
            n_fft (int): num_fft
            n_mels (int): number of mel-spectrogram bins

        Returns:
            list of tensors representing images
        """
        # TODO: make this static resource
        # inverse of mel spectrogram matrix that can revert mel-spec to power-spectrogram
        mel_basis_inv = np.matrix(librosa.filters.mel(sr, n_fft, n_mels=n_mels)).I
        # convert torch Tensor to numpy.ndarray
        imgs = imgs.cpu().detach().numpy()
        out_imgs = []
        for mel_spec in imgs:
            spec = np.dot(mel_basis_inv, mel_spec)
            # convert the power spectrum to db for better visualization
            img = librosa.power_to_db(spec, ref=np.max)
            height, width = img.shape
            out_imgs.append(torch.from_numpy(img).float().view(-1, height, width))
        return out_imgs


class SchmidtSDATrainer(NetworkTrainer):
    """Trainer for Stacked Convolutional Denoising Autoencoder."""
    def __init__(self, config: dict):
        # prepare inputs for trainer
        self.input_channel = config['input_channel']
        self.input_height = config['input_width']
        self.input_width = config['input_height']
        self.lr_init = config['lr_init']
        batch_size = config['batch_size']
        input_data_dir = config['data_dir']

        # create model
        model = SchmidtSDA(
            input_channel=self.input_channel,
            input_height=self.input_height,
            input_width=self.input_width,
        )

        dataloader_maker = VCTKLoaderMaker(input_data_dir, batch_size, use_channel=True)
        criterion = nn.MSELoss(size_average=True)
        optimizer = optim.Adam(params=model.parameters(),
                               lr=self.lr_init)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', verbose=True, factor=0.2, patience=7)

        # initialize the trainer
        super().__init__(model, dataloader_maker, criterion, optimizer,
                         epoch=config['epoch'],
                         input_size=(self.input_channel, self.input_height, self.input_width),
                         num_devices=config['num_devices'],
                         lr_scheduler=lr_scheduler)

    @staticmethod
    def input_transform(data):
        clean_img = data[0].float()
        noisy_img = data[1].float()
        return clean_img, noisy_img

    @staticmethod
    def make_performance_metric(input_, output, loss):
        pass

    @staticmethod
    def criterion_input_maker(input, output, *args, **kwargs):
        output_img, _ = output  # latent vector not used
        clean_img, _ = input
        return output_img, clean_img

    def pre_epoch_finish(self, input, output, metric_manager, train_stage):
        if train_stage == TrainStage.VALIDATE:
            # expand variables just to read easily
            clean_imgs, noisy_imgs = input
            denoised_img, _ = output
            nrow = 4
            self.add_image(clean_imgs, nrow, height=self.input_height, width=self.input_width, name='clean')
            self.add_image(noisy_imgs, nrow, height=self.input_height, width=self.input_width, name='noisy')
            self.add_image(denoised_img, nrow, height=self.input_height, width=self.input_width, name='denoised')

    def add_image(self, img, nrow, height, width, name: str):
        spec = self.make_grid_from_mel(img[:nrow, :].view(-1, 1, height, width))
        grid = torchvision.utils.make_grid(spec, nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.train_step)

    @staticmethod
    def make_grid_from_mel(imgs, sr=16000, n_fft=256, n_mels=40):  # TODO: acquire these from constants
        """
        Make image grid from a number of normalized db-scale mel-spectrograms.

        Args:
            imgs (list[torch.FloatTensor]): list of tensors representing images
            sr (int): sample rate
            n_fft (int): num_fft
            n_mels (int): number of mel-spectrogram bins

        Returns:
            list of tensors representing images
        """
        # TODO: make this static resource
        # inverse of mel spectrogram matrix that can revert mel-spec to power-spectrogram
        mel_basis_inv = np.matrix(librosa.filters.mel(sr, n_fft, n_mels=n_mels)).I

        # convert torch Tensor to numpy.ndarray
        imgs = imgs.cpu().detach().numpy()
        out_imgs = []

        # these are normalized db-scale mel-spectrograms
        for mel_spec_db_norm in imgs:
            # normalized -> denormalized db-scale mel-spec
            mel_spec = librosa.db_to_power(denormalize_db_spectrogram(mel_spec_db_norm))
            spec = np.dot(mel_basis_inv, mel_spec)
            # convert the power spectrum to db for better visualization
            img = librosa.power_to_db(spec, ref=np.max)
            height, width = img.shape
            out_imgs.append(torch.from_numpy(img).float().view(-1, height, width))
        return out_imgs


if __name__ == '__main__':
    import json
    from datasets.vctk import VCTKLoaderMaker

    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, 'config_vctk.json'), 'r') as configf:
        config = json.loads(configf.read())

    model_name = config['model_name']
    if model_name == 'dansuh':
        trainer = DansuhNetTrainer(config)
    elif model_name == 'schmidt':
        trainer = SchmidtSDATrainer(config)
    trainer.fit()
    trainer.cleanup()

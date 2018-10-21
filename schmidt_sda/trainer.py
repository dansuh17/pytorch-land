import os
import librosa
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from datasets.loader_maker import DataLoaderMaker
from utils.spectrogram import denormalize_db_spectrogram
from .schmidt_sda import SchmidtSDA
from base_trainer import NetworkTrainerOld
from tensorboardX import SummaryWriter


class SchimdtSDATrainerOld(NetworkTrainerOld):
    def __init__(self, config: dict, loadermaker_cls: DataLoaderMaker.__class__):
        super().__init__()
        self.input_root_dir = config['input_root_dir']
        self.output_root_dir = config['output_root_dir']
        self.input_data_dir = os.path.join(self.input_root_dir, config['data_dir'])
        self.log_dir = os.path.join(self.output_root_dir, config['log_dir'])
        self.model_dir = os.path.join(self.output_root_dir, config['model_dir'])

        # create output directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # parse
        self.input_width = config['input_width']
        self.input_height = config['input_height']
        self.batch_size = config['batch_size']
        self.lr_init = config['lr_init']
        self.total_epoch = config['epoch']
        self.device_ids = list(range(config['num_devices']))
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer Created')

        # create dataloaders
        loadermaker = loadermaker_cls(self.input_data_dir, self.batch_size, use_channel=True)
        self.train_dataloader = loadermaker.make_train_dataloader()
        self.val_dataloader = loadermaker.make_validate_dataloader()
        self.test_dataloader = loadermaker.make_test_dataloader()
        print('Dataloaders created')

        self.dae = SchmidtSDA(
            input_channel=1,
            input_height=self.input_height,
            input_width=self.input_width,
        ).to(self.device)

        self.dae = torch.nn.parallel.DataParallel(self.dae, device_ids=self.device_ids)
        print('Model created')
        print(self.dae)

        self.criterion = nn.MSELoss(size_average=True)
        print('Criterion created - MSE loss')

        self.optimizer = optim.Adam(params=self.dae.parameters(),
                                    lr=self.lr_init)
        print('Optimizer created - Adam')

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', verbose=True, factor=0.2, patience=7)
        print('LR scheduler created - Reduce on plateau')

        self.epoch = 0
        self.step = 0

    def train(self):
        best_loss = math.inf
        for _ in range(self.epoch, self.total_epoch):
            self.writer.add_scalar('epoch', self.epoch, self.step)

            # train - model update
            train_loss = self.run_epoch(self.train_dataloader, train=True)
            if best_loss > train_loss:
                best_loss = train_loss
                dummy_input = torch.randn(
                    (4, 1, self.input_height, self.input_width)
                ).to(self.device)
                onnx_path = os.path.join(self.model_dir, 'schmidt_sda.onnx')
                # TODO: no symbol for max_pool2d_with_indices
                # self.save_module(self.dae.module, onnx_path, save_onnx=True, dummy_input=dummy_input)

            # validate
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)
            self.epoch += 1
        # test
        self.test()

    def run_epoch(self, dataloader, train=True):
        losses = []
        dataloader_size = len(dataloader)
        for step, img_pairs in enumerate(dataloader):
            clean_img = img_pairs[0].float().to(self.device)
            noisy_img = img_pairs[1].float().to(self.device)

            output, _ = self.dae(noisy_img)  # latent vector not used
            loss = self.criterion(output, clean_img)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.step % 20 == 0:
                    loss_val = loss.item()
                    losses.append(loss_val)
                    self.log_performance(
                        self.writer, {'loss': loss_val}, self.epoch, self.step)

                if self.step % 500 == 0:  # save models
                    model_path = os.path.join(
                        self.model_dir, 'schmidt_dae_e{}.pth'.format(self.epoch))
                    self.save_module(self.dae.module, model_path)
                    self.save_module_summary(self.writer, self.dae.module, self.step)

                self.step += 1
            else:  # validation
                losses.append(loss.item())

                # save example images
                nrow = 4
                if step == dataloader_size - 1:  # save only at the end of epoch
                    self.add_image(
                        clean_img, nrow, self.input_height, self.input_width, name='clean')
                    self.add_image(
                        noisy_img, nrow, self.input_height, self.input_width, name='noisy')
                    self.add_image(
                        output, nrow, self.input_height, self.input_width, name='output')

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def add_image(self, img, nrow, height, width, name: str):
        """
        Add image grid to summary writer.

        Args:
            img (torch.FloatTensor): float tensor representation of the image
            nrow (int): number of images to be shown
            height (int): height of image (here it is the number of mel banks)
            width (int): width of single image (here this is the number of frames)
            name (str): display name
        """
        spec = self.make_grid_from_mel(img[:nrow, :].view(-1, 1, height, width))
        grid = torchvision.utils.make_grid(spec, nrow=nrow, normalize=True)
        self.writer.add_image('{}/{}'.format(self.epoch, name), grid, self.step)

    @staticmethod
    def make_grid_from_mel(imgs, sr=16000, n_fft=256, n_mels=40):  # TODO: acquire these from constants
        """
        Make image grid from a number of normalized db-scale mel-spectrograms.

        Args:
            imgs (torch.FloatTensor): tensor of images
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
        for norm_db_mel_spec in imgs:
            # denormalize + convert to power spectrogram
            mel_spec = librosa.db_to_power(denormalize_db_spectrogram(norm_db_mel_spec))
            spec = np.dot(mel_basis_inv, mel_spec)
            # convert the power spectrum to db for better visualization
            img = librosa.power_to_db(spec, ref=np.max)
            height, width = img.shape
            out_imgs.append(torch.from_numpy(img).float().view(-1, height, width))
        return out_imgs

    def validate(self):
        val_loss = self.run_epoch(self.val_dataloader, train=False)
        self.log_performance(
            self.writer, {'loss': val_loss}, self.epoch, self.step, summary_group='validation')
        return val_loss

    def test(self):
        test_loss = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f}'.format(test_loss))

    def cleanup(self):
        self.writer.close()


if __name__ == '__main__':
    import json
    from datasets.vctk import VCTKLoaderMaker

    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, 'config_vctk.json'), 'r') as configf:
        config = json.loads(configf.read())

    trainer = SchimdtSDATrainerOld(config, VCTKLoaderMaker)
    trainer.train()
    trainer.cleanup()

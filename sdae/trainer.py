import math
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from base_trainer import NetworkTrainer
from .sdae import SDAE
# from datasets.noisy_mnist import load_noisy_mnist_dataloader
from datasets.vctk import load_vctk_dataloaders
from tensorboardX import SummaryWriter


class SDAETrainer(NetworkTrainer):
    """Trainer for Stacked Denoising Auto-Encoder"""
    def __init__(self):
        super().__init__()
        self.input_root_dir = 'speech_denoise_data_in'
        self.output_root_dir = 'speech_denoise_data_out'
        self.log_dir = os.path.join(self.output_root_dir, 'tblogs')
        self.models_dir = os.path.join(self.output_root_dir, 'models')

        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.batch_size = 128
        self.num_devices = 4
        self.lr_init = 0.001
        self.end_epoch = 400
        self.device_ids = list(range(self.num_devices))

        """Uncomment one of below - 1. noisy MNIST images 2. noisy VCTK spectrograms."""
        # 1. load noisy mnist dataset
        # self.input_width = 28
        # self.input_height = 28
        # self.input_dim = 28 * 28
        # self.input_shape = (self.input_dim, )
        # self.train_dataloader, self.val_dataloader, self.test_dataloader = \
        #     load_noisy_mnist_dataloader(self.batch_size, self.input_shape)

        # 2. load noisy vctk dataset
        self.input_width = 11
        self.input_height = 40
        self.input_dim = self.input_width * self.input_height
        datapath = os.path.join(self.input_root_dir, 'vctk_processed')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            load_vctk_dataloaders(datapath, self.batch_size)
        print('Dataloader created')

        sdae = SDAE(input_dim=self.input_dim).to(self.device)
        self.sdae = torch.nn.parallel.DataParallel(sdae, device_ids=self.device_ids)
        print('Model created')
        print(self.sdae)

        self.optimizer = optim.Adam(params=self.sdae.parameters(), lr=self.lr_init)
        print('Optimizer created')

        self.summ_writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer created')

        self.criterion = nn.MSELoss()
        print('Criterion : {}'.format(self.criterion))

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', verbose=True, factor=0.2, patience=7)
        print('LR scheduler created')

        self.epoch = 0
        self.step = 0
        print('Starting from - epoch : {}, step: {}'.format(self.epoch, self.step))

    def train(self):
        """The entire training session."""
        best_loss = math.inf
        for _ in range(self.epoch, self.end_epoch):
            self.summ_writer.add_scalar('epoch', self.epoch, self.step)

            # train
            train_loss = self.run_epoch(self.train_dataloader, train=True)

            if best_loss > train_loss:
                best_loss = train_loss
                # save best model in onnx format
                dummy_input = torch.randn((10, 1, self.input_dim)).to(self.device)
                self.save_module(self.sdae.module, os.path.join(self.models_dir, 'speech_denoise.onnx'),
                                 save_onnx=True, dummy_input=dummy_input)

            # validate
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)
            self.epoch += 1
        # test
        self.test()

    def run_epoch(self, dataloader, train=True):
        """Run a single epoch of provided dataloader.
        No parameters are updated if train=False.
        """
        losses = []
        dataloader_size = len(dataloader)
        for step, img_pair in enumerate(dataloader):  # original images / corrupted images pairs
            # cast to FloatTensor (the images have been falsely converted to DoubleTensor)
            oimg = img_pair[0].float().to(self.device)
            cimg = img_pair[1].float().to(self.device)

            out = self.sdae(cimg)
            loss = self.criterion(out, oimg)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.step % 10 == 0:  # save performance metrics
                    loss_val = loss.item()
                    losses.append(loss_val)
                    self.log_performance(
                        self.summ_writer, {'loss': loss_val}, self.epoch, self.step)

                if self.step % 500 == 0:  # save models and their info
                    # save the module in onnx format
                    self.save_module(
                        self.sdae.module, os.path.join(
                            self.models_dir, 'speech_denoise_model_e{}.pth'.format(self.epoch)))
                    self.save_module_summary(
                        self.summ_writer, self.sdae, self.step, save_histogram=False)

                self.step += 1
            else:  # validation / test epoch
                losses.append(loss.item())

                # save example images
                nrow = 4
                if step == dataloader_size - 1:  # save only at the end of epoch
                    self.add_image(oimg, nrow, self.input_height, self.input_width, name='clean')
                    self.add_image(cimg, nrow, self.input_height, self.input_width, name='noisy')
                    self.add_image(out, nrow, self.input_height, self.input_width, name='output')

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def add_image(self, img, nrow, height, width, name: str):
        """
        Add image grid to summary writer.

        Args:
            img (torch.FloatTensor): float tensor representation of the image
            nrow (int): number of images to be shown
            height (int): height of single image
            width (int): width of single image
            name (str): display name
        """
        spec = self.make_grid_from_mel(img[:nrow, :].view(-1, 1, height, width))
        grid = torchvision.utils.make_grid(spec, nrow=nrow, normalize=True)
        self.summ_writer.add_image('{}/{}'.format(self.epoch, name), grid, self.step)

    @staticmethod
    def make_grid_from_mel(imgs, sr=16000, n_fft=256, n_mels=40):  # TODO: acquire these from constants
        # TODO: make this static resource
        # inverse of mel spectrogram matrix that can revert mel-spec to power-spectrogram
        mel_basis_inv = np.matrix(librosa.filters.mel(sr, n_fft, n_mels=n_mels)).I
        # convert torch Tensor to numpy.ndarray
        imgs = imgs.cpu().numpy()
        out_imgs = []
        for mel_spec in imgs:
            # convert the power spectrum to db for better visualization
            spec = np.dot(mel_basis_inv, mel_spec)
            img = librosa.power_to_db(spec, ref=np.max)
            out_imgs.append(img)
        return np.asarray(out_imgs)  # shape : num_imgs x height x width

    def test(self):
        """Test with test dataset."""
        test_loss = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f}'.format(test_loss))

    def validate(self):
        """Validation step."""
        with torch.no_grad():
            val_loss = self.run_epoch(self.val_dataloader, train=False)
            print('Epoch (validate): {:03}  Step: {:06}  Loss: {:.06f}'
                  .format(self.epoch, self.step, val_loss))
            self.summ_writer.add_scalar('validate/loss', val_loss, self.step)
        return val_loss

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.models_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'seed': self.seed,
            'model': self.sdae.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    def cleanup(self):
        self.summ_writer.close()


if __name__ == '__main__':
    # completely train mnist to stacked denoising autoencoder and cleanup afterwards
    trainer = SDAETrainer()
    trainer.train()
    trainer.cleanup()

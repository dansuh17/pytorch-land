import math
import os
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
        self.input_root_dir = 'sdae_data_in'
        self.output_root_dir = 'sdae_data_out'
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

        ### load noisy mnist dataset
        # self.input_dim = 28 * 28
        # self.input_shape = (self.input_dim, )
        # self.train_dataloader, self.val_dataloader, self.test_dataloader = \
        #     load_noisy_mnist_dataloader(self.batch_size, self.input_shape)

        ### load noisy vctk dataset
        self.input_dim = 11 * 40
        datapath = os.path.join(self.input_root_dir, 'vctk_processed')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            load_vctk_dataloaders(datapath, self.input_dim)
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
                self.save_module(self.sdae.module, os.path.join(self.models_dir, 'sdae.onnx'),
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
        for oimg, cimg in dataloader:  # original images / corrupted images pairs
            oimg, cimg = oimg.to(self.device), cimg.to(self.device)

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
                        self.sdae.module, os.path.join(self.models_dir, 'sdae_model_e{}.pth'.format(self.epoch)))
                    self.save_module_summary(
                        self.summ_writer, self.sdae, self.step, save_histogram=False)

                self.step += 1
            else:  # validation / test epoch
                losses.append(loss.item())
                # save example images
                grid_input = torchvision.utils.make_grid(cimg[:4, :].view(-1, 1, 28, 28), nrow=4, normalize=True)
                grid_output = torchvision.utils.make_grid(out[:4, :].view(-1, 1, 28, 28), nrow=4, normalize=True)
                self.summ_writer.add_image(
                    '{}/input'.format(self.epoch), grid_input, self.step)
                self.summ_writer.add_image(
                    '{}/output'.format(self.epoch), grid_output, self.step)

        avg_loss = sum(losses) / len(losses)
        return avg_loss

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

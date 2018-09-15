import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import torch.onnx
from base_trainer import NetworkTrainer
from .model import SDAE
from .noisy_dataset import load_noisy_mnist_dataloader
from tensorboardX import SummaryWriter


class SDAETrainer(NetworkTrainer):
    """Stacked Denoising Auto-Encoder"""
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ids = list(range(self.num_devices))
        self.seed = torch.initial_seed()
        print('Using random seed : {}'.format(self.seed))

        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            load_noisy_mnist_dataloader(self.batch_size)
        self.input_dim = 28 * 28
        print('Dataloader created')

        sdae = SDAE(input_dim=self.input_dim).to(self.device)
        self.sdae = torch.nn.parallel.DataParallel(sdae, device_ids=self.device_ids)
        print('Model created')
        print(self.sdae)

        self.optimizer = optim.Adam(params=self.sdae.parameters(), lr=self.lr_init)
        print('Optimizer created')

        self.summ_writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer created')

        self.criterion = nn.MSELoss(reduction='elementwise_mean')
        print('Criterion : {}'.format(self.criterion))

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', verbose=True, factor=0.2, patience=7)
        print('LR scheduler created')

        self.epoch = 0
        self.total_steps = 0
        print('Starting from - epoch : {}, step: {}'.format(self.epoch, self.total_steps))

    def train(self):
        for _ in range(self.epoch, self.end_epoch):
            self.summ_writer.add_scalar('epoch', self.epoch, self.total_steps)
            # train
            train_loss = self.run_epoch(self.train_dataloader, train=True)

            # validate
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)
            self.epoch += 1
        # test
        self.test()

    def run_epoch(self, dataloader, train=True):
        losses = []
        for oimg, cimg in dataloader:  # original images / corrupted images pairs
            oimg, cimg = oimg.to(self.device), cimg.to(self.device)

            out = self.sdae(cimg)
            loss = self.criterion(out, oimg)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.total_steps % 10 == 0:
                    loss_val = loss.item()
                    losses.append(loss_val)
                    # save performance summary
                    print('Epoch (train): {:03}  Step: {:06}  Loss: {:.6f}'
                          .format(self.epoch, self.total_steps, loss_val))
                    self.summ_writer.add_scalar('train/loss', loss_val, self.total_steps)

                if self.total_steps % 100 == 0:
                    self.save_model_summary()

                self.total_steps += 1
            else:  # validation / test epoch
                losses.append(loss.item())
                grid_input = torchvision.utils.make_grid(cimg[:4, :].view(-1, 1, 28, 28), nrow=4, normalize=True)
                grid_output = torchvision.utils.make_grid(out[:4, :].view(-1, 1, 28, 28), nrow=4, normalize=True)
                self.summ_writer.add_image(
                    '{}/input'.format(self.epoch), grid_input, self.total_steps)
                self.summ_writer.add_image(
                    '{}/output'.format(self.epoch), grid_output, self.total_steps)

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def save_model_summary(self):
        with torch.no_grad():
            for name, parameter in self.sdae.named_parameters():
                if parameter.grad is not None:
                    avg_grad = torch.mean(parameter.grad)
                    self.summ_writer.add_scalar(
                        'avg_grad/{}'.format(name), avg_grad.item(), self.total_steps)
                    self.summ_writer.add_histogram(
                        'grad/{}'.format(name), parameter.grad.cpu().numpy(), self.total_steps)

                if parameter.data is not None:
                    avg_weight = torch.mean(parameter.data)
                    self.summ_writer.add_scalar(
                        'avg_weight/{}'.format(name), avg_weight.item(), self.total_steps)
                    self.summ_writer.add_histogram(
                        'weight/{}'.format(name), parameter.data.cpu().numpy(), self.total_steps)

    def test(self):
        test_loss = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f}'.format(test_loss))

    def validate(self):
        with torch.no_grad():
            val_loss = self.run_epoch(self.val_dataloader, train=False)
            print('Epoch (validate): {:03}  Step: {:06}  Loss: {:.06f}'
                  .format(self.epoch, self.total_steps, val_loss))
            self.summ_writer.add_scalar('validate/loss', val_loss, self.total_steps)
        return val_loss

    def save_model(self, filename: str, onnx=True):
        if onnx:
            dummy_input = torch.randn((10, 1, self.input_dim))
            # TODO: set input / output names?
            torch.onnx.export(self.sdae, dummy_input, 'sdae.onnx', verbose=True)
            # TODO: check onnx validity
        else:
            model_path = os.path.join(self.models_dir, filename)
            torch.save(self.sdae, model_path)

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.models_dir, filename)
        torch.save({
            'epoch': self.epoch,
            'total_steps': self.total_steps,
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

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from datasets.loader_maker import DataLoaderMaker
from .schmidt_sda import SchmidtSDA
from base_trainer import NetworkTrainer
from tensorboardX import SummaryWriter


class SchimdtSDATrainer(NetworkTrainer):
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
            input_width=self.input_width,
            input_height=self.input_height).to(self.device)
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
            train_loss = self.run_epoch(self.test_dataloader, train=True)
            if best_loss > train_loss:
                best_loss = train_loss
                dummy_input = torch.randn(
                    (4, 1, self.input_width, self.input_height)
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
        for clean_img, noisy_img in dataloader:
            clean_img, noisy_img =\
                clean_img.float().to(self.device), noisy_img.float().to(self.device)

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
                # print 4 images in a row
                grid_input = torchvision.utils.make_grid(noisy_img[:4, :], nrow=4, normalize=True)
                grid_output = torchvision.utils.make_grid(output[:4, :], nrow=4, normalize=True)
                self.writer.add_image(
                    '{}/input'.format(self.epoch), grid_input, self.step)
                self.writer.add_image(
                    '{}/output'.format(self.epoch), grid_output, self.step)

        avg_loss = sum(losses) / len(losses)
        return avg_loss

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

    trainer = SchimdtSDATrainer(config, VCTKLoaderMaker)
    trainer.train()
    trainer.cleanup()

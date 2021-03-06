import os
import torch
from torch import nn, optim
import random
import math
from .mobilenet import MobileNet
from datasets.img_popular import load_imagenet
from tensorboardX import SummaryWriter
from base_trainer import NetworkTrainerOld


class MobileNetTrainerOld(NetworkTrainerOld):
    def __init__(self, config):
        super().__init__()
        self.input_root_dir = config['input_root_dir']
        self.output_root_dir = config['output_root_dir']
        self.log_dir = os.path.join(self.output_root_dir, config['log_dir'])
        self.models_dir = os.path.join(self.output_root_dir, config['model_dir'])

        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.num_devices = config['num_devices']
        self.batch_size = config['batch_size']
        self.lr_init = config['lr_init']
        self.end_epoch = config['epoch']
        self.image_dim = 224
        self.device_ids = list(range(self.num_devices))

        train_img_dir = os.path.join(self.input_root_dir, 'imagenet')
        self.train_dataloader, self.validate_dataloader, self.test_dataloader, misc = \
            load_imagenet(train_img_dir, self.batch_size, self.image_dim)
        self.num_classes = misc['num_classes']
        print('Dataloader created')

        net = MobileNet(self.num_classes).to(self.device)
        self.net = torch.nn.parallel.DataParallel(net, device_ids=self.device_ids)
        print('Model created')
        print(self.net)

        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.lr_init)
        print('Optimizer created')

        self.writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer created')

        self.criterion = nn.CrossEntropyLoss()
        print('Criterion : {}'.format(self.criterion))

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr']['factor'],
            patience=config['lr']['patience'],
            cooldown=config['lr']['cooldown'],
            verbose=True)
        print('LR scheduler created')

        self.epoch = 0
        self.step = 0

    def train(self):
        best_loss = math.inf
        for _ in range(self.epoch, self.end_epoch):
            self.writer.add_scalar('epoch', self.epoch, self.step)
            train_loss, _ = self.run_epoch(self.train_dataloader, train=True)
            if train_loss < best_loss:
                best_loss = train_loss
                # save the best module
                dummy_input = torch.randn((10, 3, self.image_dim, self.image_dim)).to(self.device)
                module_path = os.path.join(self.models_dir, 'mobilenet.onnx')
                self.save_module(
                    self.net.module, module_path, save_onnx=True, dummy_input=dummy_input)
            self.save_checkpoint('mobilenet_e{}_state.pth'.format(self.epoch))

            # validate
            val_loss, _ = self.validate()

            # update learning rates
            self.lr_scheduler.step(val_loss)
            self.save_learning_rate(self.writer, self.optimizer, self.step)
            self.epoch += 1
        # test step
        self.test()

    def test(self):
        test_loss, test_acc = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f} acc: {:.4f}'.format(test_loss, test_acc))

    def validate(self):
        with torch.no_grad():
            val_loss, val_acc = self.run_epoch(self.validate_dataloader, train=False)
            self.log_performance(
                self.writer,
                {'loss': val_loss, 'acc': val_acc},
                self.epoch,
                self.step)
        return val_loss, val_acc

    def run_epoch(self, dataloader, train=True):
        losses = []
        accs = []
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)

            output = self.net(imgs)
            loss = self.criterion(output, targets)

            if train:
                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # save training results
                if self.step % 20 == 0:
                    accuracy = self.calc_batch_accuracy(output, targets)
                    accs.append(accuracy.item())
                    losses.append(loss.item())
                    self.log_performance(
                        self.writer,
                        {'loss': loss.item(), 'acc': accuracy.item()},
                        self.epoch,
                        self.step)

                if self.step % 200 == 0:
                    self.save_module_summary(
                        self.writer, self.net.module, self.step)

                self.step += 1
            else:
                accuracy = self.calc_batch_accuracy(output, targets)
                accs.append(accuracy.item())
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)
        return avg_loss, avg_acc

    @staticmethod
    def calc_batch_accuracy(output, target):
        _, preds = torch.max(output, 1)
        # devide by batch size to get ratio
        accuracy = torch.sum(preds == target).float() / target.size()[0]
        return accuracy

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.models_dir, filename)
        # save the model and related checkpoints
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'torch_seed': self.seed,
            'seed': random.seed(),
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    def cleanup(self):
        self.writer.close()


if __name__ == '__main__':
    import json
    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, 'config.json'), 'r') as configf:
        config = json.loads(configf.read())

    trainer = MobileNetTrainerOld(config)
    trainer.train()
    trainer.cleanup()

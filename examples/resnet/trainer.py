import os
from resnet.model import ResNet34
import math
import torch
from torch import optim, nn
from datasets.img_popular import load_imagenet
from base_trainer import NetworkTrainerOld
from tensorboardX import SummaryWriter


class ResnetTrainerOld(NetworkTrainerOld):
    """Trainer for ResNet on imagenet 2012 dataset."""
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
        self.dataloader, self.validate_dataloader, self.test_dataloader, misc = \
            load_imagenet(train_img_dir, self.batch_size, self.image_dim)
        self.num_classes = misc['num_classes']
        print('DataLoader created')

        resnet = ResNet34(num_classes=self.num_classes, input_dim=self.image_dim).to(self.device)
        self.resnet = torch.nn.parallel.DataParallel(resnet, device_ids=self.device_ids)
        print('Model created')
        print(self.resnet)

        # Optimizer used for original paper - which doesn't train well
        # self.optimizer = optim.SGD(
        #     params=self.resnet.parameters(),
        #     lr=self.lr_init,
        #     weight_decay=0.0001,
        #     momentum=0.9
        # )
        self.optimizer = optim.Adam(params=self.resnet.parameters(), lr=self.lr_init)
        print('Optimizer created')

        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        print('Summary Writer created')

        self.criterion = nn.CrossEntropyLoss()
        print('Criterion : {}'.format(self.criterion))

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config['lr']['factor'],
            patience=10, cooldown=10, verbose=True)
        print('LR scheduler created')

        self.epoch, self.total_steps = self.set_train_status(resume=False)
        print('Starting from - Epoch : {}, Step : {}'.format(self.epoch, self.total_steps))

    def set_train_status(self, resume: bool, checkpoint_path: str=''):
        """
        Args:
            resume (bool): True if resuming previous training session
            checkpoint_path (str): path to saved checkpoint

        Returns:
            epoch (int): train epoch number
            total_steps (int): total iteration steps
        """
        if resume:
            cpt = torch.load(checkpoint_path)
            self.epoch = cpt['epoch']
            self.total_steps = cpt['total_steps']
            self.seed = cpt['seed']
            self.resnet.load_state_dict(cpt['model'])
            self.optimizer.load_state_dict(cpt['optimizer'])
            return self.epoch, self.total_steps
        return 0, 0

    def train(self):
        """The entire training session."""
        best_loss = math.inf
        for _ in range(self.epoch, self.end_epoch):
            self.summary_writer.add_scalar('epoch', self.epoch, self.total_steps)
            epoch_loss, _ = self.run_epoch(self.dataloader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                # save best module as onnx format
                dummy_input = torch.randn((10, 3, self.image_dim, self.image_dim))
                module_path = os.path.join(self.models_dir, 'resnet.onnx')
                self.save_module(
                    self.resnet.module, module_path, save_onnx=True, dummy_input=dummy_input)
            self.save_checkpoint('resnet_e{}_state.pth'.format(self.epoch))

            # validate step
            val_loss, _ = self.validate()

            # update learning rates
            self.lr_scheduler.step(val_loss)
            self.save_learning_rate(self.summary_writer, self.optimizer, self.total_steps)
            self.epoch += 1
        self.test()

    def test(self):
        test_loss, test_acc = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f} acc: {:.4f}'.format(test_loss, test_acc))

    def run_epoch(self, dataloader, train=True):
        """
        Run the model for one epoch (= full iteration) of the given data loader.

        Args:
            dataloader: loader for dataset.
            train (bool): True if performing parameter updates for the model

        Returns:
            (average_loss, average_accuracy)
        """
        losses = []
        accs = []
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(self.device), targets.to(self.device)

            # calc. the losses
            output = self.resnet(imgs)
            loss = self.criterion(output, targets)

            if train:
                # update the parameters
                self.optimizer.zero_grad()  # initialize gradients
                loss.backward()
                self.optimizer.step()

                # save training results
                if self.total_steps % 10 == 0:
                    accuracy = self.calc_batch_accuracy(output, targets)
                    accs.append(accuracy.item())
                    losses.append(loss.item())
                    self.log_performance(self.summary_writer,
                                         {'loss': loss.item(), 'acc': accuracy.item()},
                                         self.epoch,
                                         self.total_steps)

                if self.total_steps % 100 == 0:
                    self.save_module_summary(
                        self.summary_writer, self.resnet.module, self.total_steps)

                self.total_steps += 1
            else:  # no training - validation
                accuracy = self.calc_batch_accuracy(output, targets)
                accs.append(accuracy.item())
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)
        return avg_loss, avg_acc

    def validate(self):
        """
        Validate the model using the validation set.

        Returns:
            val_loss: average loss during validation
            val_acc: average accuracy during validation
        """
        with torch.no_grad():
            val_loss, val_acc = self.run_epoch(self.validate_dataloader, train=False)
            self.log_performance(self.summary_writer,
                                 {'loss': val_loss, 'acc': val_acc},
                                 self.epoch,
                                 self.total_steps,
                                 summary_group='validate')
        return val_loss, val_acc

    def save_checkpoint(self, filename: str):
        checkpoint_path = os.path.join(self.models_dir, filename)
        # save the model and related checkpoints
        torch.save({
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'seed': self.seed,
            'model': self.resnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    @staticmethod
    def calc_batch_accuracy(output, target):
        _, preds = torch.max(output, 1)
        # devide by batch size to get ratio
        accuracy = torch.sum(preds == target).float() / target.size()[0]
        return accuracy

    def cleanup(self):
        self.summary_writer.close()


if __name__ == '__main__':
    import json
    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, 'config.json'), 'r') as configf:
        config = json.loads(configf.read())

    trainer = ResnetTrainerOld(config)
    trainer.train()
    trainer.cleanup()

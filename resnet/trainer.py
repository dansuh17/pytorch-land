import os
from resnet.model import ResNet34
import math
import torch
from torch import optim, nn
from dataset import load_imagenet
from base_trainer import NetworkTrainer
from tensorboardX import SummaryWriter


class ResnetTrainer(NetworkTrainer):
    """Trainer for model."""
    def __init__(self):
        super().__init__()
        self.input_root_dir = 'resnet_data_in'
        self.output_root_dir = 'resnet_data_out'
        self.log_dir = os.path.join(self.output_root_dir, 'tblogs')
        self.models_dir = os.path.join(self.output_root_dir, 'models')
        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.num_devices = 4
        self.batch_size = 256
        self.lr_init = 0.001
        self.end_epoch = 400
        self.image_dim = 224

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ids = list([i for i in range(self.num_devices)])
        self.seed = torch.initial_seed()
        print('Using seed : {}'.format(self.seed))

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
            self.optimizer, mode='min', factor=0.1, patience=10, cooldown=30, verbose=True)
        print('LR scheduler created')

        self.epoch, self.total_steps = self.set_train_status(resume=False)
        print('Starting from - Epoch : {}, Step : {}'.format(self.epoch, self.total_steps))

    def set_train_status(self, resume: bool, checkpoint_path: str):
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
                self.save_model('resnet_model_best.pth')
            self.save_checkpoint('resnet_e{}_state.pth'.format(self.epoch))

            # validate step
            val_loss, _ = self.validate()

            # update learning rates
            self.lr_scheduler.step(val_loss)
            self.save_learning_rate()
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
                    self.save_performance_summary(loss.item(), accuracy.item())

                if self.total_steps % 100 == 0:
                    self.save_model_summary()

                self.total_steps += 1
            else:  # no training - validation
                accuracy = self.calc_batch_accuracy(output, targets)
                accs.append(accuracy.item())
                losses.append(loss.item())
                # TODO: debugging message
                print('[Debug] validation loss : {:.6f} acc : {:.6f}'
                      .format(loss.item(), accuracy.item()))

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
            self.save_performance_summary(val_loss, val_acc, summary_group='validate')
        return val_loss, val_acc

    def save_model(self, filename: str):
        model_path = os.path.join(self.models_dir, filename)
        torch.save(self.resnet, model_path)

    def save_checkpoint(self, filename: str):
        """Saves the model and training checkpoint.
        The model only saves the model, and it is usually used for inference in the future.
        The checkpoint saves the state dictionary of various modules
        required for training. Usually this information is used to resume training.
        """
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

    def save_performance_summary(self, loss, accuracy, summary_group='train'):
        print('Epoch ({}): {}\tStep: {}\tLoss: {:.6f}\tAcc: {:.6f}'
            .format(summary_group, self.epoch, self.total_steps, loss, accuracy))
        self.summary_writer.add_scalar(
            '{}/loss'.format(summary_group), loss, self.total_steps)
        self.summary_writer.add_scalar(
            '{}/accuracy'.format(summary_group), accuracy, self.total_steps)

    def save_learning_rate(self):
        """Save learning rate to summary."""
        for idx, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group['lr']
            self.summary_writer.add_scalar('lr/{}'.format(idx), lr, self.total_steps)

    def save_model_summary(self):
        with torch.no_grad():
            for name, parameter in self.resnet.named_parameters():
                if parameter.grad is not None:
                    avg_grad = torch.mean(parameter.grad)
                    # print('\tavg_grad for {} = {:.6f}'.format(name, avg_grad))
                    self.summary_writer.add_scalar(
                        'avg_grad/{}'.format(name), avg_grad.item(), self.total_steps)
                    self.summary_writer.add_histogram(
                        'grad/{}'.format(name), parameter.grad.cpu().numpy(), self.total_steps)
                if parameter.data is not None:
                    avg_weight = torch.mean(parameter.data)
                    # print('\tavg_weight for {} = {:.6f}'.format(name, avg_weight))
                    self.summary_writer.add_scalar(
                        'avg_weight/{}'.format(name), avg_weight.item(), self.total_steps)
                    self.summary_writer.add_histogram(
                        'weight/{}'.format(name), parameter.data.cpu().numpy(), self.total_steps)
        print()

    def cleanup(self):
        self.summary_writer.close()


if __name__ == '__main__':
    trainer = ResnetTrainer()
    trainer.train()
    print('Training done!')
    trainer.cleanup()

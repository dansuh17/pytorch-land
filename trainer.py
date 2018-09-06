import os
import math
import random
from model import ResNet34, ResNet32, ResNet44
import torch
from torch import optim, nn
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter


class ResnetTrainer:
    def __init__(self, dataset_name='imagenet'):
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

        self.dataset_name = dataset_name  # or 'cifar100' or 'imagenet'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ids = list([i for i in range(self.num_devices)])
        self.seed = torch.initial_seed()
        print('Using seed : {}'.format(self.seed))

        self.dataloader, self.validate_dataloader, self.test_dataloader, self.num_classes, self.image_dim \
                = self.load_dataloaders(name=self.dataset_name)
        self.validation_dataloader = None
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

        self.epoch, self.total_steps = self.set_train_status()
        print('Starting from - Epoch : {}, Step : {}'.format(self.epoch, self.total_steps))

    def load_dataloaders(self, name: str):
        if name == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_img_dir = os.path.join(self.input_root_dir, 'imagenet')
            num_classes = 1000
            image_dim = 224
            train_dataset = datasets.ImageFolder(train_img_dir, transforms.Compose([
                transforms.RandomCrop(image_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
            validate_dataset = datasets.ImageFolder(train_img_dir, transforms.Compose([
                transforms.RandomCrop(image_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
            test_dataset = datasets.ImageFolder(train_img_dir, transforms.Compose([
                transforms.CenterCrop(image_dim),
                transforms.ToTensor(),
                normalize,
            ]))
            num_data = len(dataset)
            indices = list(range(num_data))
            num_train = math.floor(num_data * 0.8)
            train_indices, valtest_indices = indices[:num_train], indices[num_train:]
            num_validate = math.floor(num_data * 0.1)
            validate_indices, test_indices = valtest_indices[:num_validate], valtest_indices[num_validate:]

            # create data loaders out of datasets
            train_dataloader = data.DataLoader(
                train_dataset,
                sampler=data.sampler.SubsetRandomSampler(train_indices),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
            validate_dataloader = data.DataLoader(
                train_dataset,
                sampler=data.sampler.SubsetRandomSampler(validate_indices),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
            test_dataloader = data.DataLoader(
                test_dataset,
                sampler=data.sampler.SubsetRandomSampler(test_indices),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
        elif name == 'cifar10':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            train_img_dir = os.path.join(self.input_root_dir, 'cifar10')
            num_classes = 10
            image_dim = 32
            dataset = datasets.CIFAR10(
                root=train_img_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_dim, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            validate_dataset = datasets.CIFAR10(
                root=train_img_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_dim, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            num_data = len(dataset)
            indices = list(range(num_data))
            random.shuffle(indices)
            num_train = math.floor(num_data * 0.9)
            train_idx, validate_idx = indices[:num_train], indices[num_train:]

            # create data loaders out of datasets
            train_dataloader = data.DataLoader(
                dataset,
                sampler=data.sampler.SubsetRandomSampler(train_idx),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
            validate_dataloader = data.DataLoader(
                dataset,
                sampler=data.sampler.SubsetRandomSampler(validate_idx),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )

            train_dataset = datasets.CIFAR10(
                root=train_img_dir, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            test_dataloader = data.DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
        elif name == 'cifar100':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            train_img_dir = os.path.join(self.input_root_dir, 'cifar100')
            num_classes = 100
            image_dim = 32
            dataset = datasets.CIFAR100(
                root=train_img_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_dim, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            validate_dataset = datasets.CIFAR100(
                root=train_img_dir, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(image_dim, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            num_data = len(dataset)
            print('Dataset size : {}'.format(num_data))
            indices = list(range(num_data))
            random.shuffle(indices)
            num_train = math.floor(num_data * 0.9)
            train_idx, validate_idx = indices[:num_train], indices[num_train:]

            # create data loaders out of datasets
            train_dataloader = data.DataLoader(
                dataset,
                sampler=data.sampler.SubsetRandomSampler(train_idx),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
            validate_dataloader = data.DataLoader(
                dataset,
                sampler=data.sampler.SubsetRandomSampler(validate_idx),
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )

            train_dataset = datasets.CIFAR10(
                root=train_img_dir, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))
            test_dataloader = data.DataLoader(
                dataset,
                pin_memory=True,
                drop_last=True,
                num_workers=4,
                batch_size=self.batch_size,
            )
        else:
            raise ValueError('Unsupported dataset : {}'.format(name))

        # split the dataset into train / validate/ test sets
        return train_dataloader, validate_dataloader, test_dataloader, num_classes, image_dim
        print('Dataset created')

    def train(self):
        for _ in range(self.epoch, self.end_epoch):
            self.summary_writer.add_scalar('epoch', self.epoch, self.total_steps)
            self.run_epoch(self.dataloader)
            self.save_checkpoint()

            val_loss, _ = self.validate()
            self.lr_scheduler.step(val_loss)
            self.save_learning_rate()
            self.epoch += 1
        test_loss, test_acc = self.run_epoch(self.test_dataloader, train=False)
        print('Test set loss: {:.6f} acc: {:.4f}'.format(test_loss, test_acc))
        print('Training complete')

    def run_epoch(self, dataloader, train=True):
        losses = []
        accs = []
        num_iters = len(dataloader.dataset)
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

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs)
        return avg_loss, avg_acc

    def set_train_status(self, resume=False):
        """
        Returns:
            epoch (int): train epoch number
            total_steps (int): total iteration steps
        """
        if resume:
            # handle the case where it resumes
            raise NotImplementedError
        return 0, 0

    def validate(self):
        with torch.no_grad():
            val_loss, val_acc = self.run_epoch(self.validate_dataloader, train=False)
            self.save_performance_summary(val_loss, val_acc, summary_group='validate')
        return val_loss, val_acc

    def calc_batch_accuracy(self, output, target):
        _, preds = torch.max(output, 1)
        # devide by batch size to get ratio
        accuracy = torch.sum(preds == target).float() / target.size()[0]
        return accuracy

    def save_performance_summary(self, loss, accuracy, summary_group='train'):
        print('Epoch: {}\tStep: {}\tLoss: {:.6f}\tAcc: {:.6f}'
            .format(self.epoch, self.total_steps, loss, accuracy))
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

    def save_checkpoint(self):
        model_path = os.path.join(
            self.models_dir, 'resnet_e{}.pkl'.format(self.epoch))
        checkpoint_path = os.path.join(
            self.models_dir, 'resnet_e{}_state.pkl'.format(self.epoch))
        # save the model and related checkpoints
        torch.save(self.resnet, model_path)
        torch.save({
            'epoch': self.epoch,
            'total_steps': self.total_steps,
            'seed': self.seed,
            'model': self.resnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self, filename: str):
        pass

    def cleanup(self):
        self.summary_writer.close()


if __name__ == '__main__':
    trainer = ResnetTrainer()
    trainer.train()

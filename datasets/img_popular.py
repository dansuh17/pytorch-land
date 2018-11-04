"""
Provides method of creating dataloaders of commonly used datasets.
All methods should commonly return a tuple of 4:
    1. train dataloader
    2. validation dataloader
    3. test dataloader
    4. any miscellaneous information / data about the dataset in dict form (None if ... well, none)
"""
import os
import math
import random
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .loader_maker import DataLoaderMaker


class MNISTLoaderMaker(DataLoaderMaker):
    """Data loader maker for MNIST dataset."""
    def __init__(self, data_root: str, batch_size: int, num_workers=4, naive_normalization=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 10
        self.num_workers = num_workers

        train_img_dir = os.path.join(data_root, 'mnist')

        # use different normalizer by option - the latter one uses the sample distribution's statistics
        normalizer = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) \
            if naive_normalization else transforms.Normalize((0.1307, ), (0.3081, ))

        # image transform (normalization)
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            normalizer,
        ])

        # create datasets
        self.train_dataset = datasets.MNIST(
            root=train_img_dir,
            train=True, download=True,
            transform=mnist_transform)
        self.validate_dataset = datasets.MNIST(
            root=train_img_dir,
            train=True, download=True,
            transform=mnist_transform)
        # create test dataset
        self.test_dataset = datasets.MNIST(
            root=train_img_dir,
            train=False, download=True,
            transform=mnist_transform)

        # split indices btwn. train and validation sets
        num_data = len(self.train_dataset)
        indices = list(range(num_data))
        random.shuffle(indices)
        num_train = math.floor(num_data * 0.9)
        self.train_idx, self.validate_idx = \
            indices[:num_train], indices[num_train:]

    def make_train_dataloader(self):
        train_dataloader = data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.train_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        return train_dataloader

    def make_validate_dataloader(self):
        validate_dataloader = data.DataLoader(
            self.validate_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.validate_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
        return validate_dataloader

    def make_test_dataloader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size,
        )
        return test_dataloader


# deprecated function
def load_mnist(data_root: str, batch_size: int):
    """Creates loaders for MNIST dataset."""
    train_img_dir = os.path.join(data_root, 'mnist')
    num_classes = 10
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_dataset = datasets.MNIST(
        root=train_img_dir,
        train=True, download=True,
        transform=mnist_transform)
    validate_dataset = datasets.MNIST(
        root=train_img_dir,
        train=True, download=True,
        transform=mnist_transform)

    # split indices btwn. train and validation sets
    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = math.floor(num_data * 0.9)
    train_idx, validate_idx = indices[:num_train], indices[num_train:]

    # create data loaders out of datasets
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=data.sampler.SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    validate_dataloader = data.DataLoader(
        validate_dataset,
        sampler=data.sampler.SubsetRandomSampler(validate_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )

    # create test dataset
    test_dataset = datasets.CIFAR10(
        root=train_img_dir,
        train=False, download=True,
        transform=mnist_transform)
    test_dataloader = data.DataLoader(
        test_dataset,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        batch_size=batch_size,
    )
    return train_dataloader, validate_dataloader, test_dataloader, {'num_classes': num_classes, 'image_dim': 28}


def load_imagenet(imagenet_dir: str, batch_size: int, image_dim=224):
    """Creates loaders of ImageNet dataset."""
    num_classes = 1000

    # image transform sequence for train and validation sets
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainval_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_dim),  # randomly resize the image
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # create datasets
    train_dataset = datasets.ImageFolder(imagenet_dir, trainval_transforms)
    validate_dataset = datasets.ImageFolder(imagenet_dir, trainval_transforms)
    test_dataset = datasets.ImageFolder(imagenet_dir, transforms.Compose([
        transforms.Resize(image_dim),  # do not adjust the aspect ratio, just resize the image
        transforms.CenterCrop(image_dim),
        transforms.ToTensor(),
        normalize,
    ]))

    # split the indices
    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
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
        batch_size=batch_size,
    )
    validate_dataloader = data.DataLoader(
        validate_dataset,
        sampler=data.sampler.SubsetRandomSampler(validate_indices),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    test_dataloader = data.DataLoader(
        test_dataset,
        sampler=data.sampler.SubsetRandomSampler(test_indices),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    return train_dataloader, validate_dataloader, test_dataloader, {'num_classes': num_classes}


def load_cifar10(data_root: str, batch_size: int, image_dim=32):
    """Creates loaders of CIFAR10 dataset."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_img_dir = os.path.join(data_root, 'cifar10')
    num_classes = 10
    trainval_transform = transforms.Compose([
        transforms.RandomCrop(image_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.CIFAR10(
        root=train_img_dir,
        train=True, download=True,
        transform=trainval_transform)
    validate_dataset = datasets.CIFAR10(
        root=train_img_dir,
        train=True, download=True,
        transform=trainval_transform)

    # split indices btwn. train and validation sets
    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = math.floor(num_data * 0.9)
    train_idx, validate_idx = indices[:num_train], indices[num_train:]

    # create data loaders out of datasets
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=data.sampler.SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    validate_dataloader = data.DataLoader(
        validate_dataset,
        sampler=data.sampler.SubsetRandomSampler(validate_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )

    # create test dataset
    test_dataset = datasets.CIFAR10(
        root=train_img_dir,
        train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,  # apply only normalization
        ]))
    test_dataloader = data.DataLoader(
        test_dataset,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        num_workers=4,
        batch_size=batch_size,
    )
    return train_dataloader, validate_dataloader, test_dataloader, {'num_classes': num_classes}


def load_cifar100(data_root: str, batch_size: int, image_dim=32):
    """Creates loaders of CIFAR100 dataset."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_img_dir = os.path.join(data_root, 'cifar100')
    num_classes = 100
    trainval_transform = transforms.Compose([
        transforms.RandomCrop(image_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.CIFAR100(
        root=train_img_dir,
        train=True, download=True,
        transform=trainval_transform)
    validate_dataset = datasets.CIFAR100(
        root=train_img_dir,
        train=True, download=True,
        transform=trainval_transform)

    # split indices btwn. train and validation sets
    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = math.floor(num_data * 0.9)
    train_idx, validate_idx = indices[:num_train], indices[num_train:]

    # create data loaders out of datasets
    train_dataloader = data.DataLoader(
        train_dataset,
        sampler=data.sampler.SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    validate_dataloader = data.DataLoader(
        validate_dataset,
        sampler=data.sampler.SubsetRandomSampler(validate_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )

    # create test dataset
    test_dataset = datasets.CIFAR100(
        root=train_img_dir,
        train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,  # apply only normalization
        ]))
    test_dataloader = data.DataLoader(
        test_dataset,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        batch_size=batch_size,
    )
    return train_dataloader, validate_dataloader, test_dataloader, {'num_classes': num_classes}

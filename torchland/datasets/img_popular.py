"""
Provides method of creating dataloaders of commonly used datasets.
All methods should commonly return a tuple of 4:
    1. train dataloader
    2. validation dataloader
    3. test dataloader
    4. any miscellaneous information / data about the dataset in dict form
        (None if ... well, none)
"""
import os
import math
import random
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .loader_builder import DataLoaderBuilder


class MNISTLoaderBuilder(DataLoaderBuilder):
    """Data loader maker for MNIST dataset."""
    def __init__(self, data_root: str, batch_size: int, num_workers=4, naive_normalization=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 10
        self.num_workers = num_workers
        self.dim = 28

        train_img_dir = os.path.join(data_root, 'mnist')

        # use different normalizer by option - the latter one uses the sample distribution's statistics
        # Be CAREFUL of sample distribution based normalization!!
        # the values will not be within boundaries of [-1, 1]
        normalizer = transforms.Normalize(mean=(0.5, ), std=(0.5, )) \
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


class LSUNLoaderBuilder(DataLoaderBuilder):
    """DataLoader maker for LSUN (Large-scale scene understanding) dataset."""
    def __init__(self, data_root: str, batch_size: int, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dim = 64  # default to 64 x 64

        img_dir = os.path.join(data_root, 'lsun')

        # image transform (normalization)
        lsun_transform = transforms.Compose([
            transforms.Resize(self.img_dim),
            transforms.CenterCrop(self.img_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # create datasets
        self.train_dataset = datasets.LSUN(
            root=img_dir,
            classes=['bedroom_train'],  # TODO: allow other classes
            transform=lsun_transform)
        self.validate_dataset = datasets.LSUN(
            root=img_dir,
            classes=['bedroom_val'],
            transform=lsun_transform)
        # create test dataset - this will share the training dataset
        self.test_dataset = datasets.LSUN(
            root=img_dir,
            classes=['bedroom_train'],
            transform=lsun_transform)

        # split indices btwn. train and validation sets
        num_data = len(self.train_dataset)
        indices = list(range(num_data))
        random.shuffle(indices)
        num_train = math.floor(num_data * 0.95)
        self.train_idx, self.test_idx = \
            indices[:num_train], indices[num_train:]

    def make_train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.train_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_validate_dataloader(self):
        return data.DataLoader(
            self.validate_dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.test_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


class ImageNetLoaderBuilder(DataLoaderBuilder):
    """
    Create a DataLoader maker for ImageNet dataset.
    This dataset is commonly used for ILSVRC (Imagenet Large Scale Visual Recognition Challenge) competition,
    and has been used in popularity since the AlexNet's success.

    The dataset commonly contains more than 1,000,000 images of 1000 classes.

    See Also:
        http://www.image-net.org/
        http://www.image-net.org/challenges/LSVRC/
    """
    def __init__(self, data_root: str, batch_size: int, num_workers: int,
                 img_dim=224, naive_normalization=False):
        super().__init__()
        self.num_classes = 1000
        self.img_dim = img_dim
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # image transform sequence for train and validation sets
        if naive_normalization:
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose([
            # randomly resize the image
            # transforms.RandomResizedCrop(self.img_dim, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(self.img_dim),
            transforms.CenterCrop(self.img_dim),
            transforms.ToTensor(),
            normalize,
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(self.img_dim),  # do not adjust the aspect ratio, just resize the image
            transforms.CenterCrop(self.img_dim),
            transforms.ToTensor(),
            normalize,
        ])

        # create datasets
        self.train_dataset = datasets.ImageFolder(self.data_root, self.train_transform)
        self.validate_dataset = datasets.ImageFolder(self.data_root, self.train_transform)
        self.test_dataset = datasets.ImageFolder(self.data_root, self.test_transform)

        # split the indices
        num_data = len(self.train_dataset)
        indices = list(range(num_data))
        random.shuffle(indices)
        # split train <-> (val + test)
        num_train = math.floor(num_data * 0.8)
        self.train_indices, valtest_indices = indices[:num_train], indices[num_train:]
        # split val <-> test
        num_validate = math.floor(num_data * 0.1)
        self.validate_indices = valtest_indices[:num_validate]
        self.test_indices = valtest_indices[num_validate:]

    def make_train_dataloader(self) -> data.DataLoader:
        # create data loaders out of datasets
        return data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.train_indices),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_validate_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.validate_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.validate_indices),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.test_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.test_indices),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


class CIFAR10LoaderBuilder(DataLoaderBuilder):
    """DataLoader maker for CIFAR-10 dataset.

    See Also: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    DEFAULT_IMG_DIM = 32

    def __init__(self, data_root: str, batch_size: int, num_workers=4,
                 img_dim=32, naive_normalization=False):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = 10
        self.img_dim = img_dim

        # data normalizers
        if naive_normalization:
            normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            normalizer = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        train_img_dir = os.path.join(data_root, 'cifar10')

        resizer = []
        # resize the image if it does not match the original
        if self.img_dim != self.DEFAULT_IMG_DIM:
            resizer.append(transforms.Resize(self.img_dim))

        trainval_transform = transforms.Compose(resizer + [
            transforms.RandomCrop(img_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer,
        ])

        test_transform = transforms.Compose(resizer + [
            transforms.CenterCrop(size=img_dim),
            transforms.ToTensor(),
            normalizer,
        ])

        self.train_dataset = datasets.CIFAR10(
            root=train_img_dir,
            train=True, download=True,
            transform=trainval_transform)
        self.validate_dataset = datasets.CIFAR10(
            root=train_img_dir,
            train=True, download=True,
            transform=trainval_transform)
        self.test_dataset = datasets.CIFAR10(
            root=train_img_dir,
            train=False, download=True,
            transform=test_transform)

        # split indices btwn. train and validation sets
        num_data = len(self.train_dataset)
        indices = list(range(num_data))
        random.shuffle(indices)
        num_train = math.floor(num_data * 0.9)
        self.train_idx, self.validate_idx = indices[:num_train], indices[num_train:]

    def make_train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.train_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_validate_dataloader(self):
        return data.DataLoader(
            self.validate_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.validate_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

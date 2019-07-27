"""
Provides methods for creating data loaders of image transfer tasks.
"""
import os
import math
import random
import itertools
import torch
from torch.utils import data
from torchvision.datasets import folder
import torchvision.transforms as transforms
from .loader_builder import DataLoaderBuilder


class Monet2PhotoDataset(data.Dataset):
    """
    Dataset containing pairs of Monet paintings and photographs.
    See CycleGAN paper: https://arxiv.org/abs/1703.10593,
    and dataset dowload script: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh
    for reference.
    """
    def __init__(self, monet_path: str, photo_path: str, transform=None, shuffle=True, random_samp: int=200000):
        """
        Args:
            monet_path(str): path to images of Monet paintings
            photo_path(str): path to photographs
            transform(callable): image transform chains
            shuffle(bool): if True, shuffles the ordering of pairs
            random_samp(int): if not None, randomly sample this amount from the entire set
        """
        super().__init__()
        self.monets = [os.path.join(monet_path, fname) for fname in os.listdir(monet_path)]
        self.photos = [os.path.join(photo_path, fname) for fname in os.listdir(photo_path)]
        self.img_loader = folder.default_loader  # default image loader provided by torchvision
        self.transform = transform

        if shuffle:
            random.shuffle(self.monets)
            random.shuffle(self.photos)

        # create all possible pairs of paths of images
        self.datapairs = list(itertools.product(self.monets, self.photos))
        if random_samp is not None and random_samp <= len(self.datapairs):
            self.datapairs = random.sample(self.datapairs, random_samp)

    def __getitem__(self, index):
        monet_path, photo_path = self.datapairs[index]

        # load images
        monet_img = self.img_loader(monet_path)
        if self.transform is not None:
            monet_img = self.transform(monet_img)

        photo_img = self.img_loader(photo_path)
        if self.transform is not None:
            photo_img = self.transform(photo_img)

        # return a pair of images - will have size: [2, ch, h, w]
        return torch.stack((monet_img, photo_img))

    def __len__(self):
        return len(self.datapairs)


class Monet2PhotoLoaderBuilder(DataLoaderBuilder):
    """
    DataLoader maker for Monet-to-Photo dataset, renown for being used by CycleGAN.
    """
    def __init__(self, batch_size: int, root_dir='./monet2photo',
                 train_monet_dir='trainA', train_photo_dir='trainB',
                 test_monet_dir='testA', test_photo_dir='testB',
                 random_samp=200000, downsize_half=False, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers

        # collect path information
        self.train_monet_path = os.path.join(root_dir, train_monet_dir)
        self.train_photo_path = os.path.join(root_dir, train_photo_dir)
        self.test_monet_path = os.path.join(root_dir, test_monet_dir)
        self.test_photo_path = os.path.join(root_dir, test_photo_dir)

        # check that all path exists
        for path in [
            self.train_monet_path,
            self.train_photo_path,
            self.train_monet_path,
            self.test_photo_path,
        ]:
            if not os.path.exists(path):
                print(f'The path: {path} does not exist!')
                raise FileNotFoundError(path)

        # downsize 256 x 256 image into half sized images
        self.img_dim = 256
        if downsize_half:
            self.img_dim = 128
        monet2photo_transform = transforms.Compose([
            transforms.Resize(size=(int(self.img_dim * 1.2), int(self.img_dim * 1.2))),
            transforms.RandomCrop(size=(self.img_dim, self.img_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.img_size = (3, self.img_dim, self.img_dim)

        # create datasets that will be loaded to each dataloader instances
        self.train_dataset = Monet2PhotoDataset(
            monet_path=self.train_monet_path,
            photo_path=self.train_photo_path,
            transform=monet2photo_transform,
            random_samp=random_samp)
        self.test_dataset = Monet2PhotoDataset(
            monet_path=self.test_monet_path,
            photo_path=self.test_photo_path,
            transform=monet2photo_transform,
            random_samp=random_samp)

        num_train_data = len(self.train_dataset)
        indices = list(range(num_train_data))
        random.shuffle(indices)
        # usually a generative model does not require a large validation set
        num_trains = math.floor(num_train_data * 0.95)
        self.train_indices, self.val_indices = \
            indices[:num_trains], indices[num_trains:]

        self.default_dataloader_kwargs = {
            'pin_memory': True,
            'drop_last': True,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
        }

    def make_train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.train_indices),
            **self.default_dataloader_kwargs)

    def make_validate_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            sampler=data.sampler.SubsetRandomSampler(self.val_indices),
            **self.default_dataloader_kwargs)

    def make_test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            shuffle=True,
            **self.default_dataloader_kwargs)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    monet2photo_dataset = Monet2PhotoDataset(
        monet_path='./cyclegan/datasets/monet2photo/trainA',
        photo_path='./cyclegan/datasets/monet2photo/trainB',
        transform=transform)

    monet2photo_loadermaker = Monet2PhotoLoaderBuilder(
        batch_size=10, root_dir='./cyclegan/datasets/monet2photo/', downsize_half=True)
    train_dataloader = monet2photo_loadermaker.make_train_dataloader()

    # print the size and length of the dataset
    print(len(train_dataloader))
    for img_pair in train_dataloader:
        print(img_pair.size())
        print(img_pair[:, 1, :, :].size())
        break

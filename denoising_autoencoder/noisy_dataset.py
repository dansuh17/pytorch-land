import os
import math
import random
import gzip
import urllib.request
from itertools import islice
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, sampler


TRAIN_IMG_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_IMG_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def to_displayable_form(img_tensor):
    return img_tensor.numpy().astype(np.uint8).resize((28, 28))


def load_noisy_mnist_dataloader(batch_size: int):
    data_root = './denoising_autoencoder/mnist'
    train_dataset = NoisyMnistDataset(data_root, train=True, zero_prob=0.25)
    # train=True because validation set is split from training dataset
    validate_dataset = NoisyMnistDataset(data_root, train=True, zero_prob=0.25)
    test_dataset = NoisyMnistDataset(data_root, train=False, zero_prob=0.30)

    num_data = len(train_dataset)
    indices = list(range(num_data))
    random.shuffle(indices)
    num_train = math.floor(num_data * 0.9)
    train_idx, validate_idx = indices[:num_train], indices[num_train:]
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler.SubsetRandomSampler(train_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        sampler=sampler.SubsetRandomSampler(validate_idx),
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        batch_size=batch_size,
    )
    return train_dataloader, validate_dataloader, test_dataloader


class NoisyMnistDataset(Dataset):
    def __init__(self, data_root: str, train=True, zero_prob=0.25):
        super().__init__()
        self.zero_prob = zero_prob
        self.image_size = 784

        # create data input path
        os.makedirs(data_root, exist_ok=True)

        if train:
            img_path = os.path.join(data_root, 'train_img.gz')
            label_path = os.path.join(data_root, 'train_label.gz')
            img_url = TRAIN_IMG_URL
            label_url = TRAIN_LABEL_URL
        else:
            img_path = os.path.join(data_root, 'test_img.gz')
            label_path = os.path.join(data_root, 'test_label.gz')
            img_url = TEST_IMG_URL
            label_url = TEST_LABEL_URL

        # load image data
        if not os.path.exists(img_path):
            print('Downloading train images: {}'.format(img_path))
            train_data_response = urllib.request.urlopen(img_url)
            with open(img_path, 'wb') as tif:
                tif.write(train_data_response.read())

        with gzip.open(img_path, 'rb') as tif:
            train_data_bytes = tif.read()[16:]
        byte_iterator = map(int, train_data_bytes)
        # slice by 784 elements and accumulate as list until it sees a blank array ([])
        self.train_data = list(iter(lambda: list(islice(byte_iterator, self.image_size)),
                                    []))

        # load label data
        if not os.path.exists(label_path):
            print('Downloading train labels')
            train_label_response = urllib.request.urlopen(label_url)
            with open(label_path, 'wb') as tlf:
                tlf.write(train_label_response.read())

        with gzip.open(label_path, 'rb') as tlf:
            train_label_bytes = tlf.read()[8:]
        self.train_label = list(map(int, train_label_bytes))

    def __getitem__(self, idx):
        # TODO: image transform
        img = np.asarray(self.train_data[idx])
        zero_mask = np.random.choice([0, 1],
                                     size=self.image_size,
                                     p=[self.zero_prob, 1 - self.zero_prob])
        img_corrupted = img * zero_mask
        return torch.FloatTensor(img), torch.FloatTensor(img_corrupted)

    def __len__(self):
        return len(self.train_data)


if __name__ == '__main__':
    dataset = NoisyMnistDataset(data_root='./denoising_autoencoder/mnist')
    print(dataset[1])

import os
import math
import random
import gzip
import urllib.request
from itertools import islice
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
from torchland.utils.noise import zero_mask_noise


TRAIN_IMG_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_IMG_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def to_displayable_form(img_tensor):
    return img_tensor.numpy().astype(np.uint8).resize((28, 28))


def load_noisy_mnist_dataloader(batch_size: int, img_shape: tuple=None):
    """Creates dataloaders with this noisy MNIST dataset."""
    data_root = './sdae/mnist'
    train_dataset = NoisyMnistDataset(data_root, img_shape, train=True, zero_prob=0.40)
    # train=True because validation set is split from training dataset
    validate_dataset = NoisyMnistDataset(data_root, img_shape, train=True, zero_prob=0.40)
    test_dataset = NoisyMnistDataset(data_root, img_shape, train=False, zero_prob=0.50)

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
    """MNIST dataset that produces pairs of images: the original 28 by 28 image
    and the same image corrupted by setting pixels to 0 with probability `zero_prob`.

    This is used to train a denoising autoencoder, so no class label is required.
    """
    def __init__(self, data_root: str, img_shape: tuple=None, train=True, zero_prob=0.25):
        super().__init__()
        self.zero_prob = zero_prob
        if img_shape is not None:
            self.img_shape = img_shape
        else:
            self.img_shape = (1, 28, 28)
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
        img = np.asarray(self.train_data[idx], dtype=np.uint8)
        # corrupt with zero mask noise
        img_corrupted = zero_mask_noise(img, zero_prob=self.zero_prob, dtype=np.uint8)

        img = img.reshape((28, 28))
        img = Image.fromarray(img, mode='L')  # mode 'L' = the array contains 0 - 255 integer values

        img_corrupted = img_corrupted.reshape((28, 28))
        img_corrupted = Image.fromarray(img_corrupted, mode='L')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
        ])
        return transform(img).view(*self.img_shape), transform(img_corrupted).view(*self.img_shape)

    def __len__(self):
        return len(self.train_data)


if __name__ == '__main__':
    dataset = NoisyMnistDataset(data_root='./sdae/mnist')
    c, o = dataset[1]
    print(c.shape)

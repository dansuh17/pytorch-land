import gzip
import urllib.request
from itertools import islice
import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_IMG_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TEST_IMG_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


def to_displayable_form(img_tensor):
    return img_tensor.numpy().astype(np.uint8).resize((28, 28))


class NoisyMnistDataset(Dataset):
    def __init__(self, train=True, zero_prob=0.5):
        super().__init__()
        self.zero_prob = zero_prob
        self.image_size = 784
        if train:
            print('Downloading train images')
            train_data_response = urllib.request.urlopen(TRAIN_IMG_URL)
            train_data_bytes = gzip.decompress(train_data_response.read())[16:]
            byte_iterator = map(int, train_data_bytes)
            # slice by 784 elements and accumulate as list until it sees a blank array ([])
            self.train_data = list(iter(lambda: list(islice(byte_iterator, self.image_size)),
                                        []))
            print('Downloading train labels')
            train_label_response = urllib.request.urlopen(TRAIN_LABEL_URL)
            train_label_bytes = gzip.decompress(train_label_response.read())[8:]
            self.train_label = list(map(int, train_label_bytes))

    def __getitem__(self, idx):
        img = np.asarray(self.train_data[idx])
        zero_mask = np.random.choice([0, 1],
                                     size=self.image_size,
                                     p=[self.zero_prob, 1 - self.zero_prob])
        img_corrupted = img * zero_mask
        return torch.FloatTensor(img), torch.FloatTensor(img_corrupted)

    def __len__(self):
        return len(self.train_data)


if __name__ == '__main__':
    dataset = NoisyMnistDataset()
    print(dataset[1])

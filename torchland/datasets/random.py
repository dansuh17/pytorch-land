from typing import Tuple
import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    """
    Dataset that generates random-valued tensor having certain shape provided.
    """
    def __init__(self, shape: Tuple[int], size: int):
        """
        :param Tuple[int] shape: tensor size that this dataset will create
        :param int size: total size of dataset
        """
        super().__init__()
        self.shape = shape
        self.size = size

    def __getitem__(self, item):
        # generate random tensor
        return torch.randn(*self.shape)

    def __len__(self):
        return self.size

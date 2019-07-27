import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .loader_builder import DataLoaderBuilder


class MixtureOfGaussiansLoaderBuilder(DataLoaderBuilder):
    """DataLoader maker for mixture of Gaussians dataset."""
    def __init__(self, total_size: int, batch_size: int, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.radius = 2
        self.num_centroids = 8
        self.std = 0.02

        # split into 8 : 1 : 1
        self.train_size = int(total_size * 0.8)
        self.test_size = int(total_size * 0.1)
        self.validate_size = self.test_size

    def make_test_dataloader(self):
        return self._create_dataloader(self.test_size)

    def make_train_dataloader(self):
        return self._create_dataloader(self.train_size)

    def make_validate_dataloader(self):
        return self._create_dataloader(self.validate_size)

    def _create_dataloader(self, data_size: int):
        return DataLoader(
            MixtureOfGaussiansDataset(
                self.radius, self.num_centroids, self.std, size=data_size),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


class MixtureOfGaussiansDataset(Dataset):
    """
    Mixture of Gaussian toy dataset.
    The centroids are equally spaced around the circle of given radius.
    The centroids are chosen as uniform manner.
    """
    def __init__(self, radius: int, num_centroids: int, std: float, size: int):
        super().__init__()
        self.num_centroids = num_centroids
        self.size = size
        self.std = std

        self.diff_angle = 2 * math.pi / num_centroids
        self.center_coordinates = radius * np.array([
            [np.cos(i * self.diff_angle),
             np.sin(i * self.diff_angle)]
            for i in range(num_centroids)
        ])

    def __getitem__(self, idx):
        # uniformly choose one of the centroids
        centroid_idx = np.random.choice(self.num_centroids)
        # sample a point from a normal distribution having mean at the chosen centroid
        sampled_point = np.random.normal(
            loc=self.center_coordinates[centroid_idx], scale=self.std)
        return torch.from_numpy(sampled_point).float()

    def __len__(self):
        return self.size


if __name__ == '__main__':
    dataset = MixtureOfGaussiansDataset(
        radius=2, num_centroids=8, std=0.02, size=100)
    for i in range(10):
        print(dataset[i])

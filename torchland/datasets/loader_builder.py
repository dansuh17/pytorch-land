from abc import ABC, abstractmethod
import math
import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class DataLoaderBuilder(ABC):
    """
    Responsible for building DataLoaders for
    train / validate / test sets of a dataset.
    """
    def __init__(self):
        pass

    @abstractmethod
    def make_train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def make_test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def make_validate_dataloader(self) -> DataLoader:
        raise NotImplementedError


class DefaultDataLoaderBuilder(DataLoaderBuilder):
    """
    Default DataLoader builder that generates train / validate / test DataLoaders
    by randomly splitting indices of a single `Dataset`.
    """
    def __init__(self, dataset: Dataset, batch_size: int, num_workers=1,
                 train_ratio=0.8, validate_ratio=0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        num_data = len(self.dataset)
        indices = list(range(num_data))
        random.shuffle(indices)  # shuffle indices inline

        # randomly split indices for each subsets
        num_train = math.floor(num_data * train_ratio)
        num_val = math.floor(num_data * validate_ratio)
        self.train_idx, valtest_idx = indices[:num_train], indices[num_train:]
        self.val_idx, self.test_idx = valtest_idx[:num_val], valtest_idx[num_val:]

    def make_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.train_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_validate_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.val_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def make_test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.test_idx),
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

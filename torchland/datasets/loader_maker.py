from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class DataLoaderBuilder(ABC):
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

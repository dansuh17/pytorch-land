from abc import ABC, abstractmethod


class DataLoaderMaker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def make_test_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def make_validate_dataloader(self):
        raise NotImplementedError


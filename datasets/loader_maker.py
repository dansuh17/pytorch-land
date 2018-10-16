class DataLoaderMaker:
    def __init__(self):
        pass

    def make_train_dataloader(self):
        raise NotImplementedError

    def make_test_dataloader(self):
        raise NotImplementedError

    def make_validate_dataloader(self):
        raise NotImplementedError


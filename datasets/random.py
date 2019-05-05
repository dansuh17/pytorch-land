from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, shape, size: int):
        super().__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

import torch
import random


class ImgHistory:
    """
    A utility class for keeping history of generated images,
    used for training CycleGAN.
    """
    def __init__(self, history_size=50):
        super().__init__()
        self.history_size = history_size
        self.data_tensors = []

    def __len__(self):
        return len(self.data_tensors)

    def get(self, img: torch.Tensor):
        if len(self.data_tensors) >= self.history_size:
            self.data_tensors = self.data_tensors[1:]  # discard the oldest
        self.data_tensors.append(img)
        # randomly choose one from history
        return random.choice(self.data_tensors)

from abc import ABC, abstractmethod
import torch
from torch import nn


class Divergence(ABC):
    @staticmethod
    @abstractmethod
    def output_activation() -> nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def conjugate_f(t: torch.Tensor):
        pass

    @classmethod
    def d_loss_func(cls, real_t, gen_t):
        return -(torch.mean(real_t) - torch.mean(cls.conjugate_f(gen_t)))

    @classmethod
    def g_loss_func(cls, gen_t):
        return -torch.mean(cls.conjugate_f(gen_t))


class GanDivergence(Divergence):
    @staticmethod
    def output_activation():
        return nn.LogSigmoid()

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return -torch.log(1 - torch.exp(t))


class KLDivergence(Divergence):
    @staticmethod
    def output_activation():
        return lambda x: x  # identity function  # TODO: make pytorch.save-able :(

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return torch.exp(t - 1)

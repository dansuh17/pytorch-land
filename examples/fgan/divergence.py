from abc import ABC, abstractmethod
import torch


class Divergence(ABC):
    @staticmethod
    @abstractmethod
    def output_activation(x: torch.Tensor) -> torch.Tensor:
        """Must provide the suggested output activation functions
        for each divergence types.
        See Table 2 from Nowozin et al.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def conjugate_f(t: torch.Tensor):
        """Must provide the 'Fenchel conjugate' function of the generator function 'f'.
        See section 2.2 "Variational Estimation of f-divergences" in Nowozin et al.
        """
        raise NotImplementedError

    @classmethod
    def d_loss_func(cls, real_t, gen_t):
        return -(torch.mean(real_t) - torch.mean(cls.conjugate_f(gen_t)))

    @classmethod
    def g_loss_func(cls, gen_t):
        return -torch.mean(cls.conjugate_f(gen_t))

    @classmethod
    def g_loss_alternative(cls, gen_t):
        """Alternative loss formulation.
        Instead of minimizing '- log E(f*(Tw(x)))', we maximize 'E(Tw(x))'.
        See section 3.2 "Practical Considerations" in Nowozin et al.
        """
        return -torch.mean(gen_t)


class KLDivergence(Divergence):
    @staticmethod
    def output_activation(x):
        return x

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return torch.exp(t - 1)


class ReverseKLDivergence(Divergence):
    @staticmethod
    def output_activation(x):
        return -torch.exp(-x)

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return -torch.log(-t) - 1


class PearsonChiSquared(Divergence):
    @staticmethod
    def output_activation(x: torch.Tensor):
        return x

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return 0.25 * (t ** 2) + t


class SquaredHellinger(Divergence):
    @staticmethod
    def output_activation(x: torch.Tensor):
        return 1 - torch.exp(-x)

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return t / (1 - t)


class JensenShannon(Divergence):
    @staticmethod
    def output_activation(x: torch.Tensor):
        return torch.log(torch.Tensor([2])).to(x.device) - torch.log(1 + torch.exp(-x))

    @staticmethod
    def conjugate_f(t: torch.Tensor):
        return -torch.log(2 - torch.exp(t))

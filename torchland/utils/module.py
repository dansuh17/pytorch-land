from torch import nn


def count_parameters(module: nn.Module):
    """
    Get number of total, trainable, and untrainable parameters.

    :param nn.Module module: module to count parameters
    :return:
        total: total number of parameters
        num_trainable: number of trainable parameters
        num_untrainable: number of untrainable parameters

    >>> conv = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, bias=False)
    >>> count_parameters(conv)
    (450, 450, 0)
    """
    num_trainable = 0
    num_untrainable = 0
    for p in module.parameters(recurse=True):
        num_p = p.numel()
        if p.requires_grad:
            num_trainable += num_p
        else:
            num_untrainable += num_p
    total = num_trainable + num_untrainable
    return total, num_trainable, num_untrainable


if __name__ == '__main__':
    import doctest
    doctest.testmod()

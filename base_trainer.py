import torch
import torch.nn as nn


class NetworkTrainer:
    """
    # TODO: REFACTOR
    1. define update step
    2. define input (perhaps in NamedTuple?)
    3. define output (perhaps in NamedTuple?)
    4. define performance metric
    5. define required hyperparameters

    input imposed -> dataloaders, trainer
    output imposed -> trainer, model
    update step -> trainer
    performance metric -> output
    hyperparameters -> trainer, dataloaders (like batch sizes)

    train(model(s), dataloader_maker, update_function, criterion(s), optimizer(s), lr_scheduler(s), **kwargs_for_trainer)

    """
    def __init__(self):
        pass


class NetworkTrainerOld:
    """Base trainer for neural net training.

    This provides a minimal set of methods that any trainer
    should implement.

    #### DEPRECATED ####
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = torch.initial_seed()
        print('Using random seed : {}'.format(self.seed))

    def test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def save_checkpoint(self, filename: str):
        """Saves the model and training checkpoint.
        The model only saves the model, and it is usually used for inference in the future.
        The checkpoint saves the state dictionary of various modules
        required for training. Usually this information is used to resume training.
        """
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError

    @staticmethod
    def save_module(module: nn.Module, path: str, save_onnx=False, dummy_input=None):
        if isinstance(module, nn.parallel.DataParallel):
            raise TypeError('Cannot save module wrapped with DataParallel!')
        if save_onnx:
            if dummy_input is None:
                raise ValueError('Must provide a valid dummy input.')
            import onnx
            # TODO: input / output names?
            # warning: cannot export DataParallel-wrapped module
            torch.onnx.export(module, dummy_input, path, verbose=True)
            # check validity of onnx IR and print the graph
            model = onnx.load(path)
            onnx.checker.check_model(model)
            onnx.helper.printable_graph(model.graph)
        else:
            torch.save(module, path)

    @staticmethod
    def performance_metric(*args, **kwargs):
        """Returns a dictionary containing various performance metrics."""
        raise NotImplementedError

    @staticmethod
    def log_performance(writer, metrics: dict, epoch: int, step: int, summary_group='train'):
        log = 'Epoch ({}): {}\tstep: {}\t'.format(summary_group, epoch, step)
        for metric_name, val in metrics.items():
            log += '{}: {:.06f}\t'.format(metric_name, val)
            # write to summary writer
            writer.add_scalar('{}/{}'.format(summary_group, metric_name), val, step)
        print(log)

    @staticmethod
    def save_learning_rate(writer, optimizer, step: int):
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            writer.add_scalar('lr/{}'.format(idx), lr, step)

    @staticmethod
    def save_module_summary(writer, module: nn.Module, step: int, save_histogram=False, verbose=False):
        # warning: saving histograms is expensive - both time and space
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if parameter.grad is not None:
                    avg_grad = torch.mean(parameter.grad)
                    if verbose:
                        print('\tavg_grad for {} = {:.6f}'.format(name, avg_grad))
                    writer.add_scalar(
                        'avg_grad/{}'.format(name), avg_grad.item(), step)
                    if save_histogram:
                        writer.add_histogram(
                            'grad/{}'.format(name), parameter.grad.cpu().numpy(), step)

                if parameter.data is not None:
                    avg_weight = torch.mean(parameter.data)
                    if verbose:
                        print('\tavg_weight for {} = {:.6f}'.format(name, avg_weight))
                    writer.add_scalar(
                        'avg_weight/{}'.format(name), avg_weight.item(), step)
                    if save_histogram:
                        writer.add_histogram(
                            'weight/{}'.format(name), parameter.data.cpu().numpy(), step)

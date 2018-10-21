import os
from collections import defaultdict
import torch
import torch.nn as nn
from datasets.loader_maker import DataLoaderMaker


class MetricManager:
    def __init__(self):
        self.metric_tracker = defaultdict(list)

    def append_metric(self, metric: dict):
        for key in metric:
            self.metric_tracker[key].append(metric[key])

    def mean(self, key: str):
        metric_list = self.metric_tracker[key]
        sum(metric_list) / len(metric_list)


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

    def __init__(self,
                 model,
                 dataloader_maker: DataLoaderMaker,
                 forward_func,
                 criterion,
                 optimizer,
                 epoch: int,
                 writer=None,
                 output_dir='data_out',
                 num_devices=1,
                 input_transform=lambda x: x,  # identity function as default
                 seed: int=None,
                 lr_scheduler=None):
        # initial settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is None:
            self.seed = torch.initial_seed()
        print('Using random seed : {}'.format(self.seed))

        # training devices to use
        self.device_ids = list(range(num_devices))

        # prepare model(s) for training
        self.model_name: str = model.__class__.__name__
        if isinstance(model, tuple):  # in case of having multiple models
            self.model = tuple(map(self.register_model, model))
        else:
            self.model = self.register_model(model)

        # create dataloaders
        self.train_dataloader = dataloader_maker.make_train_dataloader()
        self.val_dataloader = dataloader_maker.make_validate_dataloader()
        self.test_dataloader = dataloader_maker.make_test_dataloader()

        self.total_epoch = epoch
        self.forward = forward_func
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer

        # TODO: maintain a trainer state?
        self.epoch = 0
        self.train_step = 0

    def register_model(self, model):
        return torch.nn.parallel.DataParallel(
            model.to(self.device), device_ids=self.device_ids)

    def criterion_maker(self, input, output):
        raise NotImplementedError  # must be provided through __init__()

    def forward(self, model, input):
        raise NotImplementedError  # must be provided through __init__()

    @staticmethod
    def update(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def calc_loss(criterion, criterion_maker, input, output):
        return criterion(*criterion_maker(input, output))

    @staticmethod
    def lr_scheduler_metric_selector(metric):
        return metric['loss']

    def fit(self):
        for _ in range(self.epoch, self.total_epoch):
            self.writer.add_scalar('epoch', self.epoch, self.train_step)

            train_metric = self.train()
            # TODO: find the best performance metric + save onnx

            # run upon validation set
            val_metric = self.validate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(
                    self.lr_scheduler_metric_selector(val_metric))
        # run upon test set
        test_metric = self.test()

    @staticmethod
    def make_performance_metric(input, output, loss):
        return {'loss': loss}

    def post_train_step(self, input, output, metric, train_step: int):
        if train_step % 20 == 0:
            self.log_metric(self.writer, metric, self.epoch, self.train_step)

        if train_step % 500 == 0:  # save models
            model_path = os.path.join(
                self.model_dir, '{}_e{}.pth'.format(self.model_name, self.epoch))
            self.save_module(self.model.module, model_path)
            self.save_module_summary(self.writer, self.model.module, self.train_step)

    def run_epoch(self, dataloader, train=True, post_train_step=None, post_nontrain_step=None, pre_epoch_finish=None):
        metric_manager = MetricManager()
        dataloader_size = len(dataloader)
        for step, data in enumerate(dataloader):
            output = self.forward(data)
            loss = self.calc_loss(self.criterion, self.criterion_maker, data, output)

            # metric calculation
            metric = self.make_performance_metric(data, output, loss)
            metric_manager.append_metric(metric)

            if train:
                self.update(self.optimizer, loss)
                if post_train_step is not None:
                    post_train_step(data, output, metric, self.train_step)
                self.train_step += 1
            else:
                if post_nontrain_step is not None:
                    post_nontrain_step(data, output, metric, step)

            if step == dataloader_size - 1 and pre_epoch_finish is not None:
                pre_epoch_finish()
        return metric_manager

    def test(self):
        return self.run_epoch(self.test_dataloader)

    def train(self):
        # train (model update)
        return self.run_epoch(self.train_dataloader, train=True, post_train_step=self.post_train_step)

    def validate(self, post_epoch=None):
        metric = self.run_epoch(self.val_dataloader)
        if post_epoch is not None:
            post_epoch(metric, )
        return metric

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
    def log_metric(writer, metrics: dict, epoch: int, step: int, summary_group='train'):
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


#### DEPRECATED ####
class NetworkTrainerOld:
    """Base trainer for neural net training.

    This provides a minimal set of methods that any trainer
    should implement.
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

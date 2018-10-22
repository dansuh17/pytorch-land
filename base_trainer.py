import os
from abc import ABC, abstractmethod
import operator
from enum import Enum, unique
from collections import defaultdict
import torch
import torch.nn as nn
from datasets.loader_maker import DataLoaderMaker
from tensorboardX import SummaryWriter


@unique
class TrainStage(Enum):
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'


class MetricManager:
    """Class for managing multiple metrics."""
    def __init__(self):
        self.metric_tracker = defaultdict(list)

    def append_metric(self, metric: dict):
        for key in metric:
            self.metric_tracker[key].append(metric[key])

    def mean(self, key: str):
        metric_list = self.metric_tracker[key]
        return sum(metric_list) / len(metric_list)

    def mean_dict(self):
        return {key: self.mean(key) for key in self.metric_tracker.keys()}


class NetworkTrainer(ABC):
    """
    Generalized Neural Network trainer that make it easy to customize for different
    models, dataloaders, criterions, etc.
    It also provides basic logging, model saving, and summary writing.
    ``fit()`` method provides a standardized training process - train-validate-test.
    """

    def __init__(self,
                 model,
                 dataloader_maker: DataLoaderMaker,
                 criterion,
                 optimizer,
                 epoch: int,
                 input_size,
                 output_dir='data_out',
                 num_devices=1,
                 seed: int=None,
                 lr_scheduler=None):
        """
        Initialize the trainer.

        Args:
            model (nn.Module | tuple[nn.Module]): network model(s)
            dataloader_maker (DataLoaderMaker): instance creating dataloaders
            criterion: training criterion (a.k.a. loss function)
            optimizer: gradient descent optimizer
            epoch: total epochs to train (the end epoch)
            input_size (tuple[int]|tuple[tuple[int]]): size of inputs
                - must match the number of models provided
            output_dir (str): root output directory
            num_devices (int): number of GPU devices to split the batch
            seed (int): random seed to use
            lr_scheduler: learning rate scheduler
        """
        # initial settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is None:
            self.seed = torch.initial_seed()
        else:
            self.seed = seed
            torch.manual_seed(seed)
        print('Using random seed : {}'.format(self.seed))

        # training devices to use
        self.device_ids = list(range(num_devices))

        # prepare model(s) for training
        if isinstance(model, tuple):  # in case of having multiple models
            self.model_name = '_'.join(map(lambda x: x.__class__.__name__, model))
            self.model = tuple(map(self.register_model, model))
        else:
            self.model_name = model.__class__.__name__
            self.model = self.register_model(model)

        # setup and create output directories
        self.output_dir = output_dir
        self.log_dir = self._create_output_dir('logs')
        self.model_dir = self._create_output_dir('models')
        self.onnx_dir = self._create_output_dir('onnx')
        self.checkpoint_dir = self._create_output_dir('checkpoints')

        # create dataloaders
        self.train_dataloader = dataloader_maker.make_train_dataloader()
        self.val_dataloader = dataloader_maker.make_validate_dataloader()
        self.test_dataloader = dataloader_maker.make_test_dataloader()

        self.input_size = input_size  # must be torch.Size or tuple, or a tuple of them
        self.total_epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.epoch = 0
        self.train_step = 0

    @property
    def standard_metric(self):
        """Define the standard metric to compare the performance of models."""
        return 'loss'

    def _create_output_dir(self, dirname: str):
        """Creates a directory in the root output directory."""
        path = os.path.join(self.output_dir, dirname)
        os.makedirs(path, exist_ok=True)
        return path

    def save_best_model(self, prev_best_metric, curr_metric, comparison=operator.lt):
        if prev_best_metric is None:
            return curr_metric
        # compare the standard metric, and if the standard performance metric
        # is better, then save the best model
        if comparison(
                curr_metric.mean(self.standard_metric),
                prev_best_metric.mean(self.standard_metric)):
            self._save_module(save_onnx=True, prefix='best_')
            self._save_module(prefix='best')
            return curr_metric
        return prev_best_metric

    def fit(self):
        best_metric = None
        for _ in range(self.epoch, self.total_epoch):
            self.writer.add_scalar('epoch', self.epoch, self.train_step)

            train_metrics = self.train()
            # compare the train metric and save the best model - TODO: should I use the validation metric?
            best_metric = self.save_best_model(best_metric, train_metrics)

            # run upon validation set
            val_metrics = self.validate()

            # update learning rate based on validation metric
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_metrics.mean(self.standard_metric))
        # run upon test set
        test_metrics = self.test()
        print('Training complete.')

    def to_device(self, data):
        """
        Send the data to device this trainer is using

        Args:
            data (tuple|list|torch.Tensor): data

        Returns:
            device-transferred data
        """
        if isinstance(data, tuple):
            return tuple([d.to(self.device) for d in data])
        elif isinstance(data, list):
            return [d.to(self.device) for d in data]
        return data.to(self.device)

    def run_epoch(self, dataloader, train_stage: TrainStage):
        metric_manager = MetricManager()
        dataloader_size = len(dataloader)
        for step, data in enumerate(dataloader):
            data = self.to_device(self.input_transform(data))  # transform dataloader's data
            output = self.forward(self.model, data)  # feed the data to model
            loss = self.criterion(*self.criterion_input_maker(data, output))

            # metric calculation
            metric = self.make_performance_metric(data, output, loss)
            metric_manager.append_metric(metric)

            if train_stage == TrainStage.TRAIN:
                self.update(self.optimizer, loss)
                self.train_step += 1

            self.post_step(data, output, metric, train_stage=train_stage)

            # run pre-epoch-finish after the final step
            if step == dataloader_size - 1:
                self.pre_epoch_finish(data, output, metric_manager, train_stage=train_stage)
        self.on_epoch_finish(metric_manager, train_stage=train_stage)
        return metric_manager

    def test(self):
        return self.run_epoch(self.test_dataloader, TrainStage.TEST)

    def train(self):
        # train (model update)
        return self.run_epoch(self.train_dataloader, TrainStage.TRAIN)

    def validate(self):
        return self.run_epoch(self.val_dataloader, TrainStage.VALIDATE)

    def register_model(self, model):
        return torch.nn.parallel.DataParallel(
            model.to(self.device), device_ids=self.device_ids)

    @staticmethod
    def criterion_input_maker(input, output, *args, **kwargs) -> tuple:
        return output, input

    @abstractmethod
    def forward(self, model, input, *args, **kwargs):
        """
        Method for model inference.

        Args:
            model (nn.Module | tuple[nn.Module]): neural net model
            input (torch.Tensor | tuple[nn.Module]): input tensor

        Returns:
            output (torch.Tensor | tuple[nn.Module]): inference result
        """
        raise NotImplementedError

    @staticmethod
    def update(optimizer, loss):
        """
        Updates the neural network using optimization algorithm and loss gradient.

        Args:
            optimizer:
            loss:
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @staticmethod
    def make_performance_metric(input, output, loss) -> dict:
        return {'loss': loss}

    def post_step(self, input, output, metric: dict, train_stage: TrainStage):
        if train_stage == TrainStage.TRAIN:
            if self.train_step % 20 == 0:
                self.log_metric(self.writer, metric, self.epoch, self.train_step)

            if self.train_step % 500 == 0:  # save models
                self._save_module()
                self.save_module_summary(self.writer, self.model.module, self.train_step)

    def pre_epoch_finish(self, input, output, metric_manager: MetricManager, train_stage: TrainStage):
        pass

    def on_epoch_finish(self, metric_manager: MetricManager, train_stage: TrainStage):
        if train_stage == TrainStage.VALIDATE or train_stage == TrainStage.TEST:
            self.log_metric(
                self.writer,
                metric_manager.mean_dict(),
                self.epoch,
                self.train_step,
                summary_group=train_stage.value)

    def save_checkpoint(self, filename: str):
        """Saves the model and training checkpoint.
        The model only saves the model, and it is usually used for inference in the future.
        The checkpoint saves the state dictionary of various modules
        required for training. Usually this information is used to resume training.
        """
        raise NotImplementedError

    def cleanup(self):
        self.writer.close()

    @staticmethod
    def input_transform(data):
        """Provide an adapter between dataloader outputs and model inputs.
        It is an id() function by default.
        """
        return data

    def _save_module(self, save_onnx=False, prefix=''):
        if isinstance(self.model, tuple):
            models = [m.module for m in self.model]
            input_sizes = self.input_size
        else:
            models = (self.model.module, )
            input_sizes = (self.input_size, )

        for model_idx, model in enumerate(models):
            if save_onnx:
                import onnx
                # TODO: input / output names?
                # warning: cannot export DataParallel-wrapped module
                path = os.path.join(self.onnx_dir, '{}{}_onnx.pth'.format(prefix, model.__class__.__name__))
                dummy_input = torch.randn((1, ) + input_sizes[model_idx])
                torch.onnx.export(model, dummy_input, path, verbose=True)
                # check validity of onnx IR and print the graph
                model = onnx.load(path)
                onnx.checker.check_model(model)
                onnx.helper.printable_graph(model.graph)
            else:
                if prefix == '':
                    path = os.path.join(self.model_dir, 'e{:03}_{}.pth'.format(self.epoch, model.__class__.__name__))
                else:
                    path = os.path.join(self.model_dir, '{}_{}.pth'.format(prefix, model.__class__.__name__))
                torch.save(model, path)

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

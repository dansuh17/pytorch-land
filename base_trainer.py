import os
from abc import ABC, abstractmethod
import operator
import math
from enum import Enum, unique
from collections import defaultdict
import torch
import torch.nn as nn
from datasets.loader_maker import DataLoaderMaker
from tensorboardX import SummaryWriter


@unique
class TrainStage(Enum):
    """
    Enum defining each stages of training process.
    """
    TRAIN = 'train'
    VALIDATE = 'validate'
    TEST = 'test'


class MetricManager:
    """Class for managing multiple metrics."""
    def __init__(self):
        self.metric_counter = defaultdict(int)  # init to 0
        self.metric_avgs = defaultdict(float)  # init to 0.0
        self.metric_mins = defaultdict(lambda: math.inf)
        self.metric_maxes = defaultdict(lambda: -math.inf)

    def append_metric(self, metric: dict):
        """
        Introduce a new metric values and update the statistics.
        It mainly updates the count, average value, minimum value, and the maximum value.

        Args:
            metric (dict): various metric values
        """
        for key, val in metric.items():
            prev_count = self.metric_counter[key]
            prev_avg = self.metric_avgs[key]
            total_val = prev_count * prev_avg + val

            # calculate the new average
            self.metric_avgs[key] = total_val / (prev_count + 1)
            self.metric_counter[key] = prev_count + 1
            if val < self.metric_mins[key]:
                self.metric_mins[key] = val
            if val > self.metric_maxes[key]:
                self.metric_maxes[key] = val

    def mean(self, key: str) -> float:
        """
        Retrieve the mean value of the given key.

        Args:
            key (str): the key value

        Returns:
            the mean value of the key
        """
        return self.metric_avgs[key]


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
            self.model = tuple(map(self._register_model, model))
        else:
            self.model_name = model.__class__.__name__
            self.model = self._register_model(model)

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

        # save any other states or variables to maintain
        self.input_size = input_size  # must be torch.Size or tuple, or a tuple of them
        self.total_epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # initialize training process
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

    def fit(self):
        """Run the entire training process."""
        best_metric = None
        for _ in range(self.epoch, self.total_epoch):

            train_metrics = self.train()
            # compare the train metric and save the best model - TODO: should I use the validation metric?
            best_metric = self._save_best_model(best_metric, train_metrics)

            # run upon validation set
            val_metrics = self.validate()

            # update learning rate based on validation metric
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_metrics.mean(self.standard_metric))
        # run upon test set
        test_metrics = self.test()
        print('Training complete.')

    def run_epoch(self, dataloader, train_stage: TrainStage):
        self.before_epoch(train_stage=train_stage)

        metric_manager = MetricManager()  # initialize the metric manager
        dataloader_size = len(dataloader)
        for step, input_ in enumerate(dataloader):
            input_ = self._to_device(self.input_transform(input_))  # transform dataloader's data

            # perform forward and backward passes
            if train_stage == TrainStage.TRAIN:
                output = self.forward(self.model, input_)  # feed the data to model
                loss = self.criterion(*self.criterion_input_maker(input_, output))
            else:
                with torch.no_grad():  # do not accumulate gradients if not training
                    output = self.forward(self.model, input_)  # feed the data to model
                    loss = self.criterion(*self.criterion_input_maker(input_, output))

            # metric calculation
            metric = self.make_performance_metric(input_, output, loss)
            metric_manager.append_metric(metric)

            if train_stage == TrainStage.TRAIN:
                self.update(self.optimizer, loss)
                self.train_step += 1

            self.post_step(input_, output, metric, train_stage=train_stage)

            # run pre-epoch-finish after the final step
            if step == dataloader_size - 1:
                self.pre_epoch_finish(input_, output, metric_manager, train_stage=train_stage)
        self.on_epoch_finish(metric_manager, train_stage=train_stage)
        return metric_manager

    def test(self):
        return self.run_epoch(self.test_dataloader, TrainStage.TEST)

    def train(self):
        # train (model update)
        return self.run_epoch(self.train_dataloader, TrainStage.TRAIN)

    def validate(self):
        return self.run_epoch(self.val_dataloader, TrainStage.VALIDATE)

    def _register_model(self, model):
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
    def update(optimizer: torch.optim.Optimizer, loss):
        """
        Updates the neural network using optimization algorithm and loss gradient.

        Args:
            optimizer (torch.optim.Optimizer): optimizers subclassing Optimizer class
            loss: loss value
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure=None)

    @staticmethod
    def make_performance_metric(input, output, loss) -> dict:
        return {'loss': loss}

    def before_epoch(self, train_stage: TrainStage):
        # store the epoch number (to look good in tensorboard), and learning rate
        if train_stage == TrainStage.TRAIN:
            self.writer.add_scalar('epoch', self.epoch, self.train_step)
            self.save_learning_rate(self.writer, self.optimizer, self.train_step)

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
                metric_manager.metric_avgs,
                self.epoch,
                self.train_step,
                summary_group=train_stage.value)

        if train_stage == TrainStage.TRAIN:
            self.save_checkpoint()

    def save_checkpoint(self, prefix=''):
        """Saves the model and training checkpoint.
        The model only saves the model, and it is usually used for inference in the future.
        The checkpoint saves the state dictionary of various modules
        required for training. Usually this information is used to resume training.
        """
        if isinstance(self.model, tuple):
            model_state = tuple(map(lambda m: m.module.state_dict(), self.model))
            optimizer_state = tuple(map(lambda m: m.state_dict(), self.optimizer))
        else:
            model_state = self.model.module.state_dict()
            optimizer_state = self.optimizer.state_dict()

        train_state = {
            'epoch': self.epoch,
            'step': self.train_step,
            'seed': self.seed,
            'model': model_state,
            'optimizer': optimizer_state,
        }
        cptf = '{}{}_checkpoint_e{}.pth'.format(prefix, self.model_name, self.epoch)
        torch.save(train_state, os.path.join(self.checkpoint_dir, cptf))

    def resume(self, filename: str):
        cpt = torch.load(filename)
        self.seed = cpt['seed']
        torch.manual_seed(self.seed)
        self.epoch = cpt['epoch']
        self.train_step = cpt['step']

        # load the model and optimizer  # TODO: consider when they are tuples
        self.model = self.model.load_state_dict(cpt['model'])
        self.optimizer = self.optimizer.load_state_dict(cpt['optimizer'])

    def cleanup(self):
        self.writer.close()

    @staticmethod
    def input_transform(data):
        """Provide an adapter between dataloader outputs and model inputs.
        It is an id() function by default.
        """
        return data

    def _save_best_model(self, prev_best_metric, curr_metric, comparison=operator.lt):
        if prev_best_metric is None:
            return curr_metric
        # compare the standard metric, and if the standard performance metric
        # is better, then save the best model
        if comparison(
                curr_metric.mean(self.standard_metric),
                prev_best_metric.mean(self.standard_metric)):
            # onnx model saving may fail due to unsupported operators, etc.
            try:
                self._save_module(save_onnx=True, prefix='best_')
            except RuntimeError as onnx_err:
                print('Saving onnx model failed : {}'.format(onnx_err))
            self._save_module(prefix='best')
            return curr_metric
        return prev_best_metric

    def _save_module(self, save_onnx=False, prefix=''):
        if isinstance(self.model, tuple):
            # warning: cannot export DataParallel-wrapped module
            models = [m.module for m in self.model]
            input_sizes = self.input_size
        else:
            models = (self.model.module, )
            input_sizes = (self.input_size, )

        for model_idx, model in enumerate(models):
            if save_onnx:
                import onnx
                # TODO: input / output names?
                path = os.path.join(self.onnx_dir, '{}{}_onnx.pth'.format(prefix, model.__class__.__name__))
                # add batch dimension to the dummy input sizes
                dummy_input = torch.randn((1, ) + input_sizes[model_idx]).to(self.device)
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

    def _to_device(self, data):
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


#### DEPRECATED ####
class NetworkTrainerOld:
    """Base trainer for neural net training.

    This provides a minimal set of methods that any trainer
    should implement.
    """

    def __init__(self):
        print('THIS CLASS (NetworkTrainerOld) is DEPRECATED. USE NetworkTrainer INSTEAD.')
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

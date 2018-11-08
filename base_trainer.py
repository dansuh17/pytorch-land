import os
from abc import ABC, abstractmethod
import operator
import math
from enum import Enum, unique
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from datasets.loader_maker import DataLoaderMaker
from tensorboardX import SummaryWriter


@unique
class TrainStage(Enum):
    """Enum defining each stages of training process."""
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

    def set_mean(self, key: str, val):
        self.metric_avgs[key] = val


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
            optimizer (Optimizer | tuple[Optimizer]): gradient descent optimizer
            epoch: total epochs to train (the end epoch)
            input_size (tuple[int]|tuple[tuple[int]]): size of inputs
                - MUST match the number of models provided
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
            best_metric = self._save_best_model(self.model, best_metric, train_metrics)

            # run upon validation set
            val_metrics = self.validate()

            # update learning rate based on validation metric
            if self.lr_scheduler is not None:
                lr_scheduler = self._make_tuple(self.lr_scheduler)
                metrics = self._make_tuple(self.standard_metric)

                for idx, lrs in enumerate(lr_scheduler):
                    lrs.step(val_metrics.mean(self.standard_metric))

            self.epoch += 1
        # run upon test set
        test_metrics = self.test()
        print('Training complete.')

    @abstractmethod
    def run_step(self, model, criteria, optimizer, input_, train_stage: TrainStage, *args, **kwargs):
        """
        Run a single step.
        It is given all required instances for training.

        Args:
            model (nn.Module | tuple[nn.Module]): models to train
            criteria: model criteria functions
            optimizer (Optimizer | tuple[Optimizer]): model optimizers
            input_ (torch.Tensor | tuple[torch.Tensor]): inputs to models
            train_stage (TrainStage): enum indicating which stage of training it is going through

        Returns:
            (output, loss) : any combination of outputs and loss values
        """
        raise NotImplementedError

    def run_epoch(self, dataloader, train_stage: TrainStage):
        self.before_epoch(train_stage=train_stage)

        metric_manager = MetricManager()  # initialize the metric manager
        dataloader_size = len(dataloader)
        for step, input_ in enumerate(dataloader):
            input_ = self._to_device(self.input_transform(input_))  # transform dataloader's data

            # run a single step
            output, loss = self.run_step(
                self.model, self.criterion, self.optimizer, input_, train_stage)

            # metric calculation
            metric = self.make_performance_metric(input_, output, loss)
            metric_manager.append_metric(metric)

            # perform any action required after running the step
            self.post_step(input_, output, metric, train_stage=train_stage)

            if train_stage == TrainStage.TRAIN:
                self.train_step += 1

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
    def make_performance_metric(input_, output, loss) -> dict:
        return {'loss': loss.item()}

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
                self._save_all_modules()
                self._save_module_summary_all()

    def pre_epoch_finish(self, input_, output, metric_manager: MetricManager, train_stage: TrainStage):
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
        else:
            model_state = self.model.module.state_dict()

        if isinstance(self.optimizer, tuple):
            optimizer_state = tuple(map(lambda m: m.state_dict(), self.optimizer))
        else:
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

    @staticmethod
    def _make_tuple(obj):
        if isinstance(obj, tuple):
            return obj
        else:
            return obj,  # make tuple

    def _save_best_model(self, model, prev_best_metric, curr_metric, comparison=operator.lt):
        if prev_best_metric is None:
            return curr_metric

        # make tuples so that it is easier to iterate
        model = self._make_tuple(model)
        input_size = self.input_size
        standard_metric = self._make_tuple(self.standard_metric)

        # match the length
        if len(standard_metric) == 1 and len(model) != 1:
            standard_metric *= len(model)

        best_metric = prev_best_metric
        for m, std_metric, input_sz in zip(model, standard_metric, input_size):
            # compare the standard metric, and if the standard performance metric
            # is better, then save the best model
            if comparison(
                    curr_metric.mean(std_metric),
                    prev_best_metric.mean(std_metric)):
                # onnx model saving may fail due to unsupported operators, etc.
                try:
                    self._save_module(m.module, input_sz, save_onnx=True, prefix='best_')
                except RuntimeError as onnx_err:
                    print('Saving onnx model failed : {}'.format(onnx_err))
                self._save_module(m.module, input_sz, prefix='best')
                best_metric.set_mean(std_metric, curr_metric.mean(std_metric))
        return best_metric

    def _save_all_modules(self):
        models = self._make_tuple(self.model)
        input_sizes = self.input_size

        for m, input_sz in zip(models, input_sizes):
            self._save_module(m.module, input_sz)

    def _save_module(self, module, input_size: tuple, save_onnx=False, prefix=''):
        """
        Saves a single module.

        Args:
            module (nn.Module):
            input_size (tuple): input dimensions
            save_onnx (bool):
            prefix:

        Returns:

        """
        if save_onnx:
            import onnx
            # TODO: input / output names?
            path = os.path.join(self.onnx_dir, '{}{}_onnx.pth'.format(prefix, module.__class__.__name__))
            # add batch dimension to the dummy input sizes
            dummy_input = torch.randn((1, ) + input_size).to(self.device)
            torch.onnx.export(module, dummy_input, path, verbose=True)
            # check validity of onnx IR and print the graph
            model = onnx.load(path)
            onnx.checker.check_model(model)
            onnx.helper.printable_graph(model.graph)
        else:
            # note epoch for default prefix
            if prefix == '':
                prefix = 'e{:03}'.format(self.epoch)
            path = os.path.join(self.model_dir, '{}_{}.pth'.format(prefix, module.__class__.__name__))
            torch.save(module, path)

    @staticmethod
    def log_metric(writer, metrics: dict, epoch: int, step: int, summary_group='train'):
        log = 'Epoch ({}): {:03}  step: {}\t'.format(summary_group, epoch, step)
        for metric_name, val in metrics.items():
            log += '{}: {:.06f}   '.format(metric_name, val)
            # write to summary writer
            writer.add_scalar('{}/{}'.format(summary_group, metric_name), val, step)
        print(log)

    @staticmethod
    def save_learning_rate(writer, optimizer, step: int):
        if not isinstance(optimizer, tuple):
            optimizer = (optimizer, )

        for opt in optimizer:
            for idx, param_group in enumerate(opt.param_groups):
                lr = param_group['lr']
                writer.add_scalar('lr/{}'.format(idx), lr, step)

    def _save_module_summary_all(self, **kwargs):
        models = self._make_tuple(self.model)
        input_sizes = self.input_size

        for m, input_sz in zip(models, input_sizes):
            self.save_module_summary(self.writer, m.module, self.train_step, **kwargs)

    @staticmethod
    def save_module_summary(writer, module: nn.Module, step: int, save_histogram=False, verbose=False):
        # warning: saving histograms is expensive - both time and space
        module_name = module.__class__.__name__  # to distinguish among different modules
        with torch.no_grad():
            for p_name, parameter in module.named_parameters():
                if parameter.grad is not None:
                    avg_grad = torch.mean(parameter.grad)
                    if verbose:
                        print('\tavg_grad for {}_{} = {:.6f}'.format(p_name, module_name, avg_grad))
                    writer.add_scalar(
                        'avg_grad/{}_{}'.format(module_name, p_name), avg_grad.item(), step)
                    if save_histogram:
                        writer.add_histogram(
                            'grad/{}_{}'.format(module_name, p_name), parameter.grad.cpu().numpy(), step)

                if parameter.data is not None:
                    avg_weight = torch.mean(parameter.data)
                    if verbose:
                        print('\tavg_weight for {}_{} = {:.6f}'.format(module_name, p_name, avg_weight))
                    writer.add_scalar(
                        'avg_weight/{}_{}'.format(module_name, p_name), avg_weight.item(), step)
                    if save_histogram:
                        writer.add_histogram(
                            'weight/{}_{}'.format(module_name, p_name), parameter.data.cpu().numpy(), step)

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

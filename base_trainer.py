import os
import sys
import operator
from abc import ABC, abstractmethod
from typing import Dict
import math
from enum import Enum, unique
from collections import defaultdict, namedtuple
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from datasets.loader_maker import DataLoaderMaker
from tensorboardX import SummaryWriter


"""Tuple storing model information."""
if sys.hexversion >= 0x3070000:  # 'defaults' keyword appeared in ver. 3.7
    ModelInfo = namedtuple(
        'ModelInfo', ['model', 'input_size', 'metric', 'comparison'], defaults=[operator.lt])
else:
    ModelInfo = namedtuple('ModelInfo', ['model', 'input_size', 'metric', 'comparison'])
    ModelInfo.__new__.__defaults__ = (operator.lt, )


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
                 models: Dict[str, ModelInfo],
                 dataloader_maker: DataLoaderMaker,
                 criterion,
                 optimizer,
                 epoch: int,
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

        # set seed
        if seed is None:
            self.seed = torch.initial_seed()
        else:
            self.seed = seed
            torch.manual_seed(seed)
        print('Using random seed : {}'.format(self.seed))

        # training devices to use
        self.device_ids = list(range(num_devices))

        self.trainer_name = self.__class__.__name__

        # prepare model(s) for training
        self.models: Dict[str, ModelInfo] = self._validate_model_dict(models)

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
        self.total_epoch = epoch
        self.criterions = criterion
        self.optimizers = self._make_tuple(optimizer)
        if lr_scheduler is not None:
            self.lr_schedulers = self._make_tuple(lr_scheduler)
        else:
            self.lr_schedulers = None
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # initialize training process
        self.epoch = 0
        self.train_step = 0

    def _validate_model_dict(self, models: Dict[str, ModelInfo]):
        if not isinstance(models, dict):
            raise ValueError(
                'models should be an instance of dict, '
                'while provided : {}'.format(type(models)))
        for model_name in models:
            model_info = models[model_name]
            if not isinstance(model_info, ModelInfo):
                raise ValueError(
                    'Model info must have type : ' + ModelInfo.__class__.__name__)
        for model_name in models:
            model_info = models[model_name]
            model = model_info.model
            models[model_name] = model_info._replace(model=self._register_model(model))
        return models

    @staticmethod
    def _make_tuple(obj):
        """
        Make an object to tuple instance.
        If the object is already a tuple, it returns itself.

        Args:
            obj: object instance to turn into tuple

        Returns:
            tuple object
        """
        if isinstance(obj, tuple):
            return obj
        if isinstance(obj, list):
            return tuple(obj)  # make lists to tuple as well
        else:
            return (obj, )

    @staticmethod
    def _make_single_or_tuple(tuple_inst: tuple):
        if not isinstance(tuple_inst, tuple):
            raise ValueError("Input tuple_inst should be an instance of tuple!")

        if len(tuple_inst) > 1:
            return tuple_inst
        else:
            return tuple_inst[0]

    @property
    def standard_metric(self):
        """Define the standard metric to compare the performance of models."""
        return 'loss'

    def _create_output_dir(self, dirname: str):
        """Creates a directory in the root output directory."""
        path = os.path.join(self.output_dir, dirname)
        os.makedirs(path, exist_ok=True)
        return path

    def fit(self, use_val_metric=True):
        """Run the entire training process."""
        print('Using validation metric for best model : {}'.format(use_val_metric))
        best_metric = None
        for _ in range(self.epoch, self.total_epoch):
            train_metrics = self.train()

            # run upon validation set
            val_metrics = self.validate()

            # compare the metric and save the best model
            target_metric = val_metrics if use_val_metric else train_metrics
            best_metric = self._save_best_model(self.models, best_metric, target_metric)

            # update learning rate based on validation metric
            if self.lr_schedulers is not None:
                for idx, lrs in enumerate(self.lr_schedulers):
                    lrs.step(val_metrics.mean(self.standard_metric))

            self.epoch += 1
        # run upon test set
        test_metrics = self.test()
        print('Training complete.')

    @abstractmethod
    def run_step(self,
                 model: Dict[str, ModelInfo],
                 criteria,
                 optimizer,
                 input_,
                 train_stage: TrainStage,
                 *args, **kwargs):
        """
        Run a single step.
        It is given all required instances for training.

        Args:
            model (Dict[str, ModelInfo]): models to train
            criteria: model criteria functions
            optimizer (Optimizer | tuple[Optimizer]): model optimizers
            input_ (torch.Tensor | tuple[torch.Tensor]): inputs to models
            train_stage (TrainStage): enum indicating which stage of training it is going through

        Returns:
            (output, loss) : any combination of outputs and loss values
        """
        raise NotImplementedError

    def _run_epoch(self, dataloader, train_stage: TrainStage):
        """
        Runs an epoch for a dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): data loader to use in this epoch
            train_stage (TrainStage): enum indicating which training step this epoch is running

        Returns:
            metric_manager: a set of important metrics managed by the manager
        """
        self.before_epoch(train_stage=train_stage)

        metric_manager = MetricManager()  # initialize the metric manager
        dataloader_size = len(dataloader)
        for step, input_ in enumerate(dataloader):
            input_ = self._to_device(self.input_transform(input_))  # transform dataloader's data

            # run a single step
            # this step is exposed to public for custom implementation
            # the parameters passed should have equal form as passed into the constructor
            output, loss = self.run_step(
                self.models,
                self.criterions,
                self._make_single_or_tuple(self.optimizers),
                input_,
                train_stage)

            # metric calculation
            metric = self.make_performance_metric(input_, output, loss)
            metric_manager.append_metric(metric)

            # perform any action required after running the step
            self.post_step(input_, output, metric, train_stage=train_stage)

            # update train step if training
            if train_stage == TrainStage.TRAIN:
                self.train_step += 1

            # run pre-epoch-finish after the final step
            if step == dataloader_size - 1:
                self.pre_epoch_finish(input_, output, metric_manager, train_stage=train_stage)
        self.on_epoch_finish(metric_manager, train_stage=train_stage)
        return metric_manager

    def test(self):
        return self._run_epoch(self.test_dataloader, TrainStage.TEST)

    def train(self):
        # train (model update)
        return self._run_epoch(self.train_dataloader, TrainStage.TRAIN)

    def validate(self):
        return self._run_epoch(self.val_dataloader, TrainStage.VALIDATE)

    def _register_model(self, model: nn.Module):
        return torch.nn.parallel.DataParallel(
            model.to(self.device), device_ids=self.device_ids)

    @staticmethod
    def make_performance_metric(input_, output, loss) -> dict:
        return {'loss': loss.item()}

    def before_epoch(self, train_stage: TrainStage):
        # store the epoch number (to look good in tensorboard), and learning rate
        if train_stage == TrainStage.TRAIN:
            self.writer.add_scalar('epoch', self.epoch, self.train_step)
            self.save_learning_rate(self.writer, self.optimizers, self.train_step)

    def post_step(self, input, output, metric: dict, train_stage: TrainStage):
        if train_stage == TrainStage.TRAIN:
            if self.train_step % 20 == 0:
                self.log_metric(self.writer, metric, self.epoch, self.train_step)

            if self.train_step % 500 == 0:  # save models
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

        # save all modules at the end of epoch
        self._save_all_modules()

    def save_checkpoint(self, prefix=''):
        """Saves the model and training checkpoint.
        The model only saves the model, and it is usually used for inference in the future.
        The checkpoint saves the state dictionary of various modules
        required for training. Usually this information is used to resume training.
        """
        # get state_dict for all models
        model_state = {}
        for model_name in self.models:
            model_state[model_name] = self.models[model_name].model.state_dict()

        optimizer_state = tuple(map(lambda m: m.state_dict(), self.optimizers))

        train_state = {
            'epoch': self.epoch,
            'step': self.train_step,
            'seed': self.seed,
            'models': model_state,  # tuple of states (even for len == 1)
            'optimizers': optimizer_state,  # tuple of states
        }
        cptf = '{}{}_checkpoint_e{}.pth'.format(prefix, self.trainer_name, self.epoch)
        torch.save(train_state, os.path.join(self.checkpoint_dir, cptf))

    def resume(self, filename: str):
        """
        Load checkpoint file and set internal fields accordingly.

        Args:
            filename (str): file path to checkpoint file
        """
        cpt = torch.load(filename)
        self.seed = cpt['seed']
        torch.manual_seed(self.seed)
        self.epoch = cpt['epoch']
        self.train_step = cpt['step']

        # load the model and optimizer
        model_state = cpt['models']
        for model_name in self.models:
            model_info = self.models[model_name]
            model = model_info.model
            # replace ModelInfo's field
            self.models[model_name] = model_info._replace(
                model=model.load_state_dict(model_state[model_name]))

        self.optimizers = [o.load_state_dict(state_dict)
                           for o, state_dict in zip(self.optimizers, cpt['optimizers'])]

    def cleanup(self):
        self.writer.close()

    @staticmethod
    def input_transform(data):
        """Provide an adapter between dataloader outputs and model inputs.
        It is an id() function by default.
        """
        return data

    def _save_best_model(self, models: Dict[str, ModelInfo], prev_best_metric, curr_metric):
        if prev_best_metric is None:
            return curr_metric

        best_metric = prev_best_metric
        for model_info in models.values():
            model = model_info.model
            compare_metric = model_info.metric
            comparison = model_info.comparison
            input_size = model_info.input_size

            # compare the standard metric, and if the standard performance metric
            # is better, then save the best model
            if comparison(
                    curr_metric.mean(compare_metric),
                    prev_best_metric.mean(compare_metric)):
                # onnx model saving may fail due to unsupported operators, etc.
                try:
                    self._save_module(model.module, input_size, save_onnx=True, prefix='best_')
                except RuntimeError as onnx_err:
                    print('Saving onnx model failed : {}'.format(onnx_err))
                self._save_module(model.module, input_size, prefix='best')
                best_metric.set_mean(compare_metric, curr_metric.mean(compare_metric))
        return best_metric

    def _save_all_modules(self):
        for model_info in self.models.values():
            self._save_module(model_info.model.module,
                              model_info.input_size)

    def _save_module(self, module, input_size: tuple, save_onnx=False, prefix=''):
        """
        Saves a single module.

        Args:
            module (nn.Module): module to be saved
            input_size (tuple): input dimensions
            save_onnx (bool): save in ONNX format if True
            prefix: prefix for file name
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
    def save_learning_rate(writer, optimizers, step: int):
        for opt in optimizers:
            for idx, param_group in enumerate(opt.param_groups):
                lr = param_group['lr']
                writer.add_scalar('lr/{}'.format(idx), lr, step)

    def _save_module_summary_all(self, **kwargs):
        for model_info in self.models.values():
            self._save_module_summary(self.writer,
                                      model_info.model.module,
                                      self.train_step,
                                      **kwargs)

    @staticmethod
    def _save_module_summary(writer, module: nn.Module, step: int,
                             save_histogram=False, verbose=False):
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

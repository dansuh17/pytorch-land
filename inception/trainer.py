import os
import torch
from .inception_v2 import InceptionV2
from datasets.img_popular import load_imagenet
from base_trainer import NetworkTrainer


class InceptionNetTrainer(NetworkTrainer):
    def __init__(self, config):
        super().__init__()
        self.input_root_dir = config['input_root_dir']
        self.output_root_dir = config['output_root_dir']
        self.log_dir = os.path.join(self.output_root_dir, config['log_dir'])
        self.models_dir = os.path.join(self.output_root_dir, config['model_dir'])

        # create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.num_devices = config['num_devices']
        self.batch_size = config['batch_size']
        self.lr_init = config['lr_init']
        self.end_epoch = config['epoch']
        self.image_dim = 229
        self.device_ids = list(range(self.num_devices))

        train_img_dir = os.path.join(self.input_root_dir, 'vctk_preprocess')
        self.train_dataloader, self.validate_dataloader, self.test_dataloader, misc = \
            load_imagenet(train_img_dir, self.batch_size, self.image_dim)
        self.num_classes = misc['num_classes']
        print('Dataloader created')

        net = InceptionV2(self.num_classes, self.image_dim)
        self.net = torch.nn.parallel.DataParallel(net, device_ids=self.device_ids)
        print('Model created')
        print(self.net)


if __name__ == '__main__':
    import json
    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, 'config.json'), 'r') as configf:
        config = json.loads(configf.read())
    trainer = InceptionNetTrainer(config)

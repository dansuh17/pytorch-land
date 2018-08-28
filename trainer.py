from model import ResNet34
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ResnetTrainer:
    def __init__(self):
        self.train_img_dir = 'resnet_data_in'
        self.image_dim = 224
        self.resnet = ResNet34(num_classes=1000)
        self.dataset = datasets.ImageFolder(self.train_img_dir, transforms.Compose([
            transforms.CenterCrop(self.image_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    def validate(self):
        pass

    def train_one_epoch(self):
        pass

    def train(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def cleanup(self):
        pass
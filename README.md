# PyTorch Land

This is a repository of select implementations of deep learning models using [pytorch](https://pytorch.org/). 

There is also my own mini-framework for model training 
(which I call simply `NetworkTrainer`), that ended up with 
something very similar to [ignite](https://pytorch.org/ignite/), 
created in order to reduce common boilerplating.
More flexible and intuitive than ignite, in my opinion :).

More models to be added, and improvements on `NetworkTrainer` is under way.

## Installation

```bash
pip3 install pytorch-land
```

See [pypi page](https://pypi.org/project/pytorch-land/) for package details.

## Implemented Models

### CNN architectures
- ResNet (2015) [[paper](https://arxiv.org/abs/1512.03385)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/resnet)] 
- Inception v.2 (2015) [[paper](https://arxiv.org/abs/1512.00567)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/inception)]
- MobileNet (2017) [[paper](https://arxiv.org/abs/1704.04861)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/mobilenet)]

### GANs
- GAN (2014) [[paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/gan)]
- DCGAN (2015) [[paper](https://arxiv.org/abs/1511.06434)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/dcgan)]
- InfoGAN (2016) [[paper](https://arxiv.org/pdf/1606.03657.pdf)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/infogan)]
- f-GAN (2016) [[paper](https://arxiv.org/abs/1606.00709)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/fgan)]
- UnrolledGAN (2016) [[paper](https://arxiv.org/abs/1611.02163)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/unrolled_gan)] [[train-examples](https://github.com/deNsuh/pytorch-land/blob/master/unrolled_gan/unrolledgan_train_results.ipynb)]
- ACGAN (2016) [[paper](https://arxiv.org/abs/1610.09585)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/acgan)] [[train-examples](https://github.com/deNsuh/pytorch-land/tree/master/acgan)]
- BEGAN (2017) [[paper](https://arxiv.org/abs/1703.10717)] [[code & examples](https://github.com/deNsuh/pytorch-land/tree/master/began)]
- CycleGAN (2017) [[paper](https://arxiv.org/abs/1703.10593)] [[code & examples](https://github.com/deNsuh/pytorch-land/tree/master/cyclegan)]

### Autoencoders
- Stacked Denoising Autoencoders [[paper](https://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/sdae)]
- Stacked Convolutional Denoising Autoencoders (2017) [[paper](https://mediatum.ub.tum.de/doc/1381852/54858742554.pdf)] [[code](https://github.com/deNsuh/pytorch-land/tree/master/schmidt_sda)]

## Requirements

Required packages are specified in [requirements.txt](https://github.com/deNsuh/pytorch-land/blob/master/requirements.txt)
file. The packages can be installed using the following command:

```bash
pip3 install -r requirements.txt
```

Notably, the codes are compatible with **pytorch 0.4** - working on with **pytorch 1.1** compatibility.

## NetworkTrainer

## Datasets

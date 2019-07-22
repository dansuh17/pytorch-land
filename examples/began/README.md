# BEGAN training results on MNIST dataset

- Even with MNIST dataset, the value of gamma (what I call equilibrium constant) does influence the results significantly
- One must carefully choose gamma value.
- With large gamma value as 0.7, the generator does produce diverse images, but they also look sloppy. Numbers __6 and 8__, __3 and 8__, __1 and 7__ sometimes cannot be distinguished.
- The model uses `ReLU` and `LeakyReLU` instead of `ELU` as stated in the paper, but the difference was negligible.


## Model

The model follows the model depicted in _Figure 1_ in the [original paper](https://arxiv.org/abs/1703.10717), except the dimensions of the image is
`(28 x 28 x 1)` for MNIST image instead of `(32 x 32 x 3)`.

![model](https://user-images.githubusercontent.com/14329563/52031206-bd06be00-255e-11e9-8a90-4a967732ee85.png)

## Setup
- gamma (what I call the equilibrium constant): 0.7
- batch size: 32

## Training Examples

- epoch 0

![image](https://user-images.githubusercontent.com/14329563/52031284-140c9300-255f-11e9-86ad-f82db1b7a8f8.png)

- epoch 5

![image](https://user-images.githubusercontent.com/14329563/52031291-1ff85500-255f-11e9-82ca-691ea93766f1.png)

- epoch 10

![image](https://user-images.githubusercontent.com/14329563/52031313-30a8cb00-255f-11e9-8d32-213e903e6d61.png)

- epoch 20

![image](https://user-images.githubusercontent.com/14329563/52031324-40281400-255f-11e9-8f4b-5a96a550ce0b.png)

- epoch 50

![image](https://user-images.githubusercontent.com/14329563/52031328-44543180-255f-11e9-9f53-010e30a3b823.png)


## Loss function graphs

- With discriminator loss converging to value close to 0.011 generator loss converging to value close to 0.008, it seems the model is trained consistently with equilibrium constant (gamma) of 0.7.

- Discriminator loss

![d-loss](https://user-images.githubusercontent.com/14329563/52031349-5e8e0f80-255f-11e9-976e-41ef5b610af2.png)

- Generator loss

![g-loss](https://user-images.githubusercontent.com/14329563/52031363-74033980-255f-11e9-8002-c802a36c3b41.png)

## Convergence Measure

- convergence measure for BEGAN is defined as:

![convergence-measure-def](https://user-images.githubusercontent.com/14329563/52031540-5edada80-2560-11e9-8da7-0dd34d124532.png)

- The convergence measure also converges as generated images look more authentic.

![convergence-measure](https://user-images.githubusercontent.com/14329563/52031496-2d620f00-2560-11e9-952d-0b28c2b4f6bb.png)

## k

![image](https://user-images.githubusercontent.com/14329563/52031528-508cbe80-2560-11e9-9ccc-5f0a48dcf57b.png)


## Failure cases
- larger batch size (of around 128) failed to generate any image and failes to learn

![image](https://user-images.githubusercontent.com/14329563/52031566-7fa33000-2560-11e9-89d6-6af66d9baac1.png)

- too small gamma value (0.3) made the generator easily fall into mode-collapse, while generating more genuine images

![image](https://user-images.githubusercontent.com/14329563/52031612-be38ea80-2560-11e9-9e60-bd12984526b8.png)

![image](https://user-images.githubusercontent.com/14329563/52031599-ae210b00-2560-11e9-9a0f-cec5d690143c.png)

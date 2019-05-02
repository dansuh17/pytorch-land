# CycleGAN

- **paper**: [Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"](https://arxiv.org/abs/1703.10593)

![cyclegan-model](../assets/cyclegan_model.png)


### Total loss
- total loss consists of two GAN losses (each for generator) and the cycle-consistency loss

![total-loss](../assets/total_loss.png)

### Cycle-consistency loss
- tells the generators to reduce reconstruction errors

![cycle_consistency_loss](../assets/cycle_consistency_loss.png)

## Examples
- trained on monet2photo dataset

![cyclegan_examples](../assets/cyclegan_examples.png)


## Failure Cases (CycleGAN Deception)
- CycleGAN learns to hide information in order to reduce reconstruction loss.
- See [my blog post](http://densuh.github.io/jekyll/update/2019/04/24/cyclegan-deception.html) about CycleGAN's deception tricks and the to fix it.
- The fix is to add gaussian noise onto the generated image before reconstructing with the other generator.
- Here are some examples of failing generation, but successful reconstruction is done.

#### monet -> photo -> monet
![bad_monet_reconst](../assets/bad-monet-reconstruct.png)

#### photo -> monet -> photo
![bad_photo_reconst](../assets/bad-photo-reconstruct.png)

### References On CycleGAN Deception
- [reddit on cyclegan mode collapse](https://www.reddit.com/r/MachineLearning/comments/b0a7qq/d_cyclegan_model_collapse_any_bright_ideas/)

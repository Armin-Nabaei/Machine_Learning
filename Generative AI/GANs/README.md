
## Adversary in Generative Adversarial Network comes from competition between the generator and the discriminator.

![Alt text](<Screenshot 2023-12-23 at 3.07.11 PM.png>)
Improved stability of two GANs models in the Image Generation Domain, which suffer from vanishing gradients in an early training stage:
- train generator while freezing discriminator
- The stability of the GAN game suffers if you have sparse gradients.
- LeakyReLU is  Good in both generator and discriminator.
- For downsampling, use Average pooling, conv2d + stride.
- For upsampling, use pixel Shuffle, convtransposed2d + stride.
_____________________________________________________
![Alt text](<Screenshot 2023-12-23 at 3.13.58 PM.png>)
_____________________________________________________
# Cycle_GANs
## "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
## https://arxiv.org/abs/1703.10593

![Alt text](<Screenshot 2023-12-23 at 2.16.42 PM.png>)
![Alt text](<Screenshot 2023-12-23 at 2.17.01 PM.png>)

### CycleGAN for Face-mask Remover Application:

Loss = Adversarial Loss + Cycle Consistency Loss

It uses instance normalization in which instead of normalizing whole batches of specific channels in batch normalization, it normalizes one batch.

In patch GANs, a Discriminator is a discriminator for generative adversarial, which only penalizes structure at the scale of local image patches. The PatchGANs discriminator tries to classify if each N×N patch in an image is real or fake.

It means that at the end of Discriminator, instead of binary classification, which expresses fake or real, we have a matrix of values in which each element represents a local patch of images.

CycleGANs create unpaired images (it has the capability of generating images from different domains). The two main properties of this model are loss and normalization functions, in which the Loss includes Adversarial loss + cycle consistency loss. 

Cycle Consistency Loss is a comparison between the reversed process of creating real input from fake output by real input, in which the computed value is considered more important than the adversarial loss value. Also, it uses Instance normalization instead of batch normalization to process batches one by one, not all in one step.

The utilized discriminator in CycleGANs is named PatchGANs with the specification of dividing the image structures into N patches and penalizing the scale of local image patches, so the output will be a matrix of values in which each element represents a local patch of images.

__________________________________________________

# Pro_GANs 

## https://arxiv.org/abs/1710.10196

### "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
![Alt text](<Screenshot 2023-12-23 at 2.31.58 PM.png>)

### ProGAN for Image Generation Application:

The implemented technique in ProGANs for a training process includes multiple sub-training regarding feeding new high-resolution layers. This property avoids high shock in a well-trained model. Another novelty is introducing a minibatch discrimination layer at the final block to compute features across mini-batches. Despite WGANs in which the discriminator is trained 5 times and the generator 1 time, here, discriminator and generator training times are equal.

The technique used in this method is starting training from low-resolution input and output, then feeding one by one higher resolution layers to the model. This addition, during training, is repeated in a smooth manner. It avoids high shock in a well-trained model.

Another technique is introducing minibatch discrimination to compute features across mini-batches.
In WGANs, the training discriminator to the generator is 5 to 1, but in ProGANs is 1 to 1 per epoch.

Regularizer is added to WGANs loss to keep weights updating away from zero.
The final conv block includes a minibatch std to add one channel to output channels. All conv layers use non-linear activation (LReLU), and a fully-connected layer, which uses linear activation (No activation), and the linear  activation for conv is “tanh.”

__________________________________________________


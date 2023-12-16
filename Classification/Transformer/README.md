# Multi Cross-Scale Mix Local And Non-Local Fusion Activation Maps for Attention-Modules in Image Classification
## "Hybrid Model consists of Transformer & CNN with results of 98.55% in CIFAR10, 90.16% in CIFAR100, 90.16% in Stanford Cars, and , FOOD-101: 90.04 datasets.

### Download my trained weights links:
#### CIFAR-10 : 
https://drive.google.com/drive/folders/15sN8wzAYOH6XFQiAGzA2Zg13Ykb4QZad?usp=sharing

#### CIFAR-100 : 
https://drive.google.com/file/d/1Bn1DEB7KqOCSBmu2e0HyMW99GXfPYy92/view?usp=sharing

#### Stanford Cars:
https://drive.google.com/file/d/1Bn1DEB7KqOCSBmu2e0HyMW99GXfPYy92/view?usp=sharing
___________________________

<img width="504" alt="image" src="https://github.com/arminn84/Machine-Learning/assets/150948007/72efa21e-2263-4335-a545-654905f209eb">

As we know, the attention mechanism is the cornerstone of human perception. The Perceptron system of vision captures the richest features of resources by processing knowledge distillation. Another aspect of the visual structure is parallel computing, which makes it amazingly fast and robust. The above factors inspired researchers to add attention mechanisms in inductive learning models.
An Attention-Module consists of two consecutive modules, self-attention and MLP, feeding by patches. A transformer model may include stack transformer layers, each composed of multi-head self-attention. They can be exploited on top of a CNN’s backbone or as a pure model, and when they apply in the encoder-encoder model, called ‘Transformer.’
This work comprises a hybrid model based on mining mix of local-global information and feature refinement. Our Results demonstrated the superiority of our proposed model among state-of-the-art papers by reaching: 98.55% in CIFAR10, and 90.16% in CIFAR100.

# Model 
<img width="431" alt="image" src="https://github.com/arminn84/Machine-Learning/assets/150948007/f3d41ed9-abb9-40f1-81b9-3c751ce66312">

## Challenges, Motivation, and Objectives

Models based on CNN’s layers mainly suffer from small receptive fields which only operate on the local neighborhood of corresponding data, so they work very well in primary layers due to their inductive bias property. Still, for the rest of the framework, regarding the lack of ability to gather global context and lack of global dependencies, researchers utilize transformer models based on the self-attention mechanism by eliminating the below shortages:
Processing long-range sequences for hierarchical inputs
More accessibility to higher representation levels 
Long-time and huge computation complexity in pre-training and need for large data sets.
Degradation information in the last layers and collapsing attention mechanism in terms of creating identical feature maps.

---
ViT is a starving model that needs a massive dataset for training, such as millions of samples. Minorizing the structure, using refinement feature maps inside the model, and feeding the model with multiscale maps instead of the original size have encouraged researchers. And to reduce computation by novelties in low-rank factorization methods, such as choosing low-dimensional projection modalities instead of linear projection layers.

---

The central part of this proposed work is creating a light-weighted generic family of attention-based models for defining interaction and dependencies between pixels along and across channels to find coarse and fine-grained object classes in images with negligible computation cost. This work aims to obtain the richest input data representation for image classification tasks.

---

The central part of this proposed work is creating a light-weighted generic family of attention-based models for defining interaction and dependencies between pixels along and across channels to find coarse and fine-grained object classes in images with negligible computation cost. The rest of proposing methods include:
Increasing the embedding dimension to avoid attention collapse
 mutual connection for better learning attention weights
 exploiting other types of convolutions to enhance receptive fields 
utilized more connections to refine feature activation maps
 replacing feed-forward layers to reduce the number of parameters which leads to decreased computational complexity
Designing mixed local-global attention models
considering an appropriate backbone to extract features


-------------------
# Proposed Modules

<img width="416" alt="image" src="https://github.com/arminn84/Machine-Learning/assets/150948007/719ab48c-22c7-40a9-8863-3473cb698f0e">
<img width="292" alt="image" src="https://github.com/arminn84/Machine-Learning/assets/150948007/b7fa866d-1938-45d2-a883-b582711e8162">


# Proposed Bottleneck Structure
<img width="416" alt="image" src="https://github.com/arminn84/Machine-Learning/assets/150948007/4683274b-ccb9-4aa2-955d-cf805442166b">

### We made the transformer module deeper and higher in dimension by adding our unique bottleneck structure to explore broader representation by grouping linear transformation layers.

### We group input along the channel axis; each group is convolved with a separate filter/kernel, and at the end, the input and output will be concatenated (Depth wise Convolution).
___________________________________________________________________
















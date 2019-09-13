# Pytorch EfficientNet, with pretrained weights

A single-file, pythonic, modularized implementation of EfficientNet 
as introduced in
[\[Tan & Le 2019\]: EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

#### Parameters
 
* ***b***, *(int)* - Model index, e.g. 1 for EfficientNet-B1.
* ***in_channels***, *(int)*, *(Default=3)* - Number of channels in input image.
* ***n_classes***, *(int)*, *(Default=1000)* - Number of 
output classes.
* ***input_spatial_shape***, *(int or iterable of ints)*, 
*(Default=None)* - Spatial dimensionality of input image, e.g
 (height, width) or single integer *size* for shape (*size*, *size*). 
If None, default image shape will be used for each model index. 
* ***activation***, *(callable)*, 
*(Default=Swish())* - Activation function.
* ***bias***, *(bool)*, 
*(Default=False)* - Whether to enable bias in convolution operations.
* ***drop_connect_rate***, *(float)*, 
*(Default=0.2)* - DropConnect rate, set to 0 to disable DropConnect.
* ***override_dropout_rate***, *(float or None)*, 
*(Default=None)* - override default dropout rate with this value,
if *None* is provided, the default dropout rate will be used 
for each model index.
* ***bn_epsilon***, *(float)*, 
*(Default=0.001)* - Batch normalizaton epsilon.
* ***bn_momentum***, *(float)*, 
*(Default=0.01)* - Batch normalization momentum.
* ***pretrained***, *(bool)*, 
*(Default=False)* - Whether to initialize model with weights 
pretrained on ImageNet dataset
* ***progress***, *(bool)*, 
*(Default=False)* - Show progress bar when downloading 
pretrained weights.

The default parameter values are the ones that were used in 
[original implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

#### Evaluation
A simple script to evaluate pretrained models against Imagenet 
validation set is provided in [this notebook](imagenet_eval.ipynb).

Validation set accuracies achieved by this implementation against ImageNet
dataset:

| Model | Accuracy, % |
| --- | --- |
| B0 | 76.43% |
| B1 | 78.396% |
| B2 | 79.804% |
| B3 | 81.542% |
| B4 | 83.036% |
| B5 | 83.79% |
| B6 | 84.136% |
| B7 | 84.578% |

#### Coming soon
\>\> A sample training script
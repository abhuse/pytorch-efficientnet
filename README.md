# Pytorch EfficientNet, with pretrained weights

A single-file, pythonic, modularized implementation of EfficientNet 
as introduced in
[\[Tan & Le 2019\]: EfficientNet: Rethinking Model Scaling
for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

#### Usage
The example below creates an EfficientNet-B0
model that takes 3-channel image of shape \[256, 320\] (height, width)
as input and outputs distribution over 50 classes, 
model weights are initialized
with weights pretrained on ImageNet dataset:
```python
from efficientnet import EfficientNet
model = EfficientNet(b=0,
                  in_channels=3,
                  n_classes=50,
                  in_spatial_shape=(256,320),
                  pretrained=True
                  )
# x - tensor of shape [batch_size, in_channels, image_height, image_width]

 # to get predictions:
 pred = model(x) 
 
 # to extract features:
 features = model.get_features(x)
 ```
```pred``` is now a tensor containing output logits while 
```features``` is a list of 7 tensors representing outputs
of each of the EfiicientNet's 7 intermediate convolutional stages.

If you would like to experiment with this implementation
(e.g. try out different hyper-parameters) feel free to 
run [cifar100_train.ipynb](cifar100_train.ipynb) which contains a 
baseline CIFAR-100 training routine.

 
#### Parameters
 
* ***b***, *(int)* - Model index, e.g. 1 for EfficientNet-B1.
* ***in_channels***, *(int)*, *(Default=3)* - Number of channels in input image.
* ***n_classes***, *(int)*, *(Default=1000)* - Number of 
output classes.
* ***in_spatial_shape***, *(int or iterable of ints)*, 
*(Default=None)* - Spatial dimensionality of input image, tuple
 (height, width) or single integer *size* for shape (*size*, *size*). 
If None, default image shape will be used for each model index. 
* ***activation***, *(callable)*, 
*(Default=Swish())* - Activation function.
* ***bias***, *(bool)*, 
*(Default=False)* - Whether to enable bias in convolution operations.
* ***drop_connect_rate***, *(float)*, 
*(Default=0.2)* - DropConnect rate, set to 0 to disable DropConnect.
* ***dropout_rate***, *(float or None)*, 
*(Default=None)* - override default dropout rate with this value,
if *None* is provided, the default dropout rate will be used.
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
validation set is provided in [imagenet_eval.ipynb](imagenet_eval.ipynb).

Accuracy achieved by models with pre-trained weights against ImageNet
dataset:

| Model | Accuracy, % |
| --- | --- |
| EfficientNet-B0 | 76.43% |
| EfficientNet-B1 | 78.396% |
| EfficientNet-B2 | 79.804% |
| EfficientNet-B3 | 81.542% |
| EfficientNet-B4 | 83.036% |
| EfficientNet-B5 | 83.79% |
| EfficientNet-B6 | 84.136% |
| EfficientNet-B7 | 84.578% |

# EfficientNetV2 EfficientNetV1 in Pytorch with pretrained weights

A single-file implementation of EfficientNetV2 and EfficientNetV1 as introduced in:  
[\[Tan & Le 2021\]: EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf)  
[\[Tan & Le 2019\]: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Pretrained Weights
Original implementations of both [EfficientNetV2](https://github.com/google/automl/tree/master/efficientnetv2) and [EfficientNetV1](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) include pretrained weigths in Tensorflow format. 
These weigths were converted to Pytorch format and are provided in this repository.

## Accuracy

### EfficientNet V2
| Model | ImageNet 1k Top-1 accuracy, % |
| --- | --- |
| EfficientNetV2-b0 | 77.590% |
| EfficientNetV2-b1 | 78.872% |
| EfficientNetV2-b2 | 79.388% |
| EfficientNetV2-b3 | 82.260% |
| EfficientNetV2-S | 84.282% |
| EfficientNetV2-M | 85.596% |
| EfficientNetV2-L | 86.298% |
| EfficientNetV2-XL | 86.414% |

### EfficientNet V1
| Model | ImageNet 1k Top-1 Accuracy, % |
| --- | --- |
| EfficientNet-B0 | 76.43% |
| EfficientNet-B1 | 78.396% |
| EfficientNet-B2 | 79.804% |
| EfficientNet-B3 | 81.542% |
| EfficientNet-B4 | 83.036% |
| EfficientNet-B5 | 83.79% |
| EfficientNet-B6 | 84.136% |
| EfficientNet-B7 | 84.578% |

## Usage

Check out  [cifar100_train.ipynb](cifar100_train.ipynb) if you would like to experiment with models.  
To evaluate pretrained models against Imagenet validation set, run [imagenet_eval.ipynb](imagenet_eval.ipynb).

### EfficientNet V2
The example below creates an EfficientNetV2-S model that takes 3-channel image of shape [224, 224]
as input and outputs distribution over 50 classes, model weights are initialized with weights pretrained on ImageNet
dataset:
```python
import torch
from efficientnet_v2 import EfficientNetV2

model = EfficientNetV2('s',
                        in_channels=3,
                        n_classes=50,
                        pretrained=True)

# x - tensor of shape [batch_size, in_channels, image_height, image_width]
x = torch.randn([10, 3, 224, 224])

# to get predictions:
pred = model(x) 
print('out shape:', pred.shape)
# >>> out shape: torch.Size([10, 50])

# to extract features:
features = model.get_features(x)
for i, feature in enumerate(features):
    print('feature %d shape:' % i, feature.shape)
# >>> feature 0 shape: torch.Size([10, 48, 56, 56])
# >>> feature 1 shape: torch.Size([10, 64, 28, 28])
# >>> feature 2 shape: torch.Size([10, 160, 14, 14])
# >>> feature 3 shape: torch.Size([10, 256, 7, 7])
 ```

### EfficientNet (V1, original)

The example below creates an EfficientNet-B0 model that takes 3-channel image of shape [224, 224]
as input and outputs distribution over 50 classes, model weights are initialized with weights pretrained on ImageNet
dataset:

```python
import torch
from efficientnet import EfficientNet

model = EfficientNet(b=0,
                  in_channels=3,
                  n_classes=50,
                  in_spatial_shape=(224,224),
                  pretrained=True
                  )

# x - tensor of shape [batch_size, in_channels, image_height, image_width]
x = torch.randn([10, 3, 224, 224])

# to get predictions:
pred = model(x) 
print('out shape:', pred.shape)
#  >>> out shape: torch.Size([10, 50])

# to extract features:
features = model.get_features(x)
for i, feature in enumerate(features):
    print('feature %d shape:' % i, feature.shape)
# >>> feature 0 shape: torch.Size([10, 16, 112, 112])
# >>> feature 1 shape: torch.Size([10, 24, 56, 56])
# >>> feature 2 shape: torch.Size([10, 40, 28, 28])
# >>> feature 3 shape: torch.Size([10, 80, 14, 14])
# >>> feature 4 shape: torch.Size([10, 112, 14, 14])
# >>> feature 5 shape: torch.Size([10, 192, 7, 7])
# >>> feature 6 shape: torch.Size([10, 320, 7, 7])
 ```


## Parameters


### EfficientNet V2
* ***model_name***, *(str)* - Model name, one of 'b0', 'b1', 'b2', 'b3', 's', 'm', 'l', 'xl'
* ***in_channels***, *(int)*, *(Default=3)* - Number of channels in input image
* ***n_classes***, *(int)*, *(Default=1000)* - Number of output classes
* ***tf_style_conv***, *(bool)*, *(Default=False)* - Whether to simulate "SAME" padding of Tensorflow's convolution op. Set to *True* when evaluating pretrained models against Imagenet dataset
* ***in_spatial_shape***, *(int or iterable of ints)*,
  *(Default=None)* - Spatial dimensionality of input image, tuple
  (height, width) or single integer *size* for shape (*size*, *size*). 
  It is recommended to specify this parameter only when *tf_style_conv=True*
* ***activation***, *(str)*, *(Default='silu')* - Activation function  
* ***activation_kwargs***, *(dict)*, *(Default=None)* - Keyword arguments to pass to activation function
* ***bias***, *(bool)*,
  *(Default=False)* - Enable bias in convolution operations
* ***drop_connect_rate***, *(float)*,
  *(Default=0.2)* - DropConnect rate, set to 0 to disable DropConnect
* ***dropout_rate***, *(float or None)*,
  *(Default=None)* - Dropout rate, set to *None* to use default dropout rate for each model
* ***bn_epsilon***, *(float)*,
  *(Default=0.001)* - Batch normalizaton epsilon
* ***bn_momentum***, *(float)*,
  *(Default=0.01)* - Batch normalization momentum
* ***pretrained***, *(bool)*,
  *(Default=False)* - Initialize model with weights pretrained on ImageNet dataset
* ***progress***, *(bool)*,
  *(Default=False)* - Show progress bar when downloading pretrained weights

The default parameter values are the ones that were used in
[original implementation](https://github.com/google/automl/tree/master/efficientnetv2).


### EfficientNet V1
* ***b***, *(int)* - Model index, e.g. 1 for EfficientNet-B1
* ***in_channels***, *(int)*, *(Default=3)* - Number of channels in input image
* ***n_classes***, *(int)*, *(Default=1000)* - Number of output classes
* ***in_spatial_shape***, *(int or iterable of ints)*,
  *(Default=None)* - Spatial dimensionality of input image, tuple
  (height, width) or single integer *size* for shape (*size*, *size*). If None, default image shape will be used for
  each model index
* ***activation***, *(callable)*,
  *(Default=Swish())* - Activation function
* ***bias***, *(bool)*,
  *(Default=False)* - Enable bias in convolution operations
* ***drop_connect_rate***, *(float)*,
  *(Default=0.2)* - DropConnect rate, set to 0 to disable DropConnect
* ***dropout_rate***, *(float or None)*,
  *(Default=None)* - Dropout rate, set to *None* to use default dropout rate for each model
* ***bn_epsilon***, *(float)*,
  *(Default=0.001)* - Batch normalizaton epsilon
* ***bn_momentum***, *(float)*,
  *(Default=0.01)* - Batch normalization momentum
* ***pretrained***, *(bool)*,
  *(Default=False)* - Initialize model with weights pretrained on ImageNet dataset
* ***progress***, *(bool)*,
  *(Default=False)* - Show progress bar when downloading pretrained weights

The default parameter values are the ones that were used in
[original implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).


## Requirements

* Python v3.5+
* Pytorch v1.0+
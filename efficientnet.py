from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs
from torch.utils import model_zoo


def _pair(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return (x, x)


class SamePaddingConv2d(nn.Conv2d):
    def __init__(self,
                 input_spatial_shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 **kwargs):
        super(SamePaddingConv2d, self).__init__(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                dilation=dilation,
                                                **kwargs)

        self.input_spatial_shape = _pair(input_spatial_shape)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        in_height, in_width = self.input_spatial_shape
        filter_height, filter_width = kernel_size
        stride_heigth, stride_width = stride
        dilation_height, dilation_width = dilation

        out_height = int(ceil(float(in_height) / float(stride_heigth)))
        out_width = int(ceil(float(in_width) / float(stride_width)))

        pad_along_height = max((out_height - 1) * stride_heigth +
                               filter_height + (filter_height - 1) * (dilation_height - 1) - in_height, 0)
        pad_along_width = max((out_width - 1) * stride_width +
                              filter_width + (filter_width - 1) * (dilation_width - 1) - in_width, 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        if any(p > 0 for p in paddings):
            self.zero_pad = nn.ZeroPad2d(paddings)
        else:
            self.zero_pad = None

        self.out_spatial_shape = (out_height, out_width)

    def check_input(self, x):
        if x.size(2) != self.input_spatial_shape[0] or \
                x.size(3) != self.input_spatial_shape[1]:
            raise ValueError("input spatial shape mismatch")

    def forward(self, x):
        self.check_input(x)

        if self.zero_pad is not None:
            x = self.zero_pad(x)

        x = super(SamePaddingConv2d, self).forward(x)
        return x


class Swish(nn.Module):
    def __init__(self,
                 beta=1.0,
                 beta_learnable=False):
        super(Swish, self).__init__()

        if beta == 1.0 and not beta_learnable:
            self._op = self.simple_swish
        else:
            self.beta = nn.Parameter(torch.full([1], beta),
                                     requires_grad=beta_learnable)
            self._op = self.advanced_swish

    def simple_swish(self, x):
        return x * torch.sigmoid(x)

    def advanced_swish(self, x):
        return x * torch.sigmoid(self.beta * x)

    def forward(self, x):
        return self._op(x)


class DropConnect(nn.Module):
    def __init__(self, rate=0.5):
        super(DropConnect, self).__init__()
        self.keep_prob = None
        self.set_rate(rate)

    def set_rate(self, rate):
        if not 0 <= rate < 1:
            raise ValueError("rate must be 0<=rate<1, got {} instead".format(rate))
        self.keep_prob = 1 - rate

    def forward(self, x):
        if self.training:
            random_tensor = self.keep_prob + torch.rand([x.size(0), 1, 1, 1],
                                                        dtype=x.dtype,
                                                        device=x.device)
            binary_tensor = torch.floor(random_tensor)
            return torch.mul(torch.div(x, self.keep_prob), binary_tensor)
        else:
            return x


class SqueezeExcitate(nn.Module):
    def __init__(self,
                 in_channels,
                 se_size,
                 activation=None):
        super(SqueezeExcitate, self).__init__()
        self.dim_reduce = nn.Conv2d(in_channels=in_channels,
                                    out_channels=se_size,
                                    kernel_size=1)
        self.dim_restore = nn.Conv2d(in_channels=se_size,
                                     out_channels=in_channels,
                                     kernel_size=1)
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dim_reduce(x)
        x = self.activation(x)
        x = self.dim_restore(x)
        x = torch.sigmoid(x)
        return torch.mul(inp, x)


class MBConvBlock(nn.Module):
    def __init__(self,
                 input_spatial_shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expansion_factor,
                 bn_epsilon=None,
                 bn_momentum=None,
                 se_size=None,
                 activation=None,
                 drop_connect_rate=None,
                 bias=False):
        """
        Initialize new MBConv block
        :param input_spatial_shape: image shape, e.g. [height, width] or int size for [size, size]
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for depth-wise convolution
        :param stride: stride for depth-wise convolution
        :param expansion_factor: expansion factor
        :param bn_epsilon: batch normalization epsilon
        :param bn_momentum: batch normalization momentum
        :param se_size: number of features in reduction layer of Squeeze-and-Excitate layer
        :param activation: activation function
        :param drop_connect_rate: DropConnect rate
        :param bias: enable bias in convolution operations
        """
        super(MBConvBlock, self).__init__()

        if se_size is not None and se_size < 1:
            raise ValueError("se_size must be >=1, got {} instead".format(se_size))

        if drop_connect_rate is not None and not 0 <= drop_connect_rate < 1:
            raise ValueError("drop_connect_rate must be in range [0,1), got {} instead".format(drop_connect_rate))

        if not (isinstance(expansion_factor, int) and expansion_factor >= 1):
            raise ValueError("expansion factor must be int and >=1, got {} instead".format(expansion_factor))

        exp_channels = in_channels * expansion_factor
        kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
        stride = [stride] * 2 if isinstance(stride, int) else stride

        bn_kwargs = {}
        if bn_epsilon is not None:
            bn_kwargs["eps"] = bn_epsilon
        if bn_momentum is not None:
            bn_kwargs["momentum"] = bn_momentum

        # expansion convolution
        if expansion_factor != 1:
            self.expansion_conv_enabled = True
            self.expand_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=exp_channels,
                                         kernel_size=(1, 1),
                                         bias=bias)
            self.expand_bn = nn.BatchNorm2d(exp_channels, **bn_kwargs)
        else:
            self.expansion_conv_enabled = False

        # depth-wise convolution
        self.dp_conv = SamePaddingConv2d(
            input_spatial_shape=input_spatial_shape,
            in_channels=exp_channels,
            out_channels=exp_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=exp_channels,
            bias=bias)
        self.dp_bn = nn.BatchNorm2d(exp_channels, **bn_kwargs)

        self.out_spatial_shape = self.dp_conv.out_spatial_shape

        if se_size is not None:
            self.se_enabled = True
            self.se = SqueezeExcitate(exp_channels, se_size, activation=activation)
        else:
            self.se_enabled = False

        if drop_connect_rate is not None:
            self.drop_connect_enabled = True
            self.drop_connect = DropConnect(drop_connect_rate)
        else:
            self.drop_connect_enabled = False

        if in_channels == out_channels and all(s == 1 for s in stride):
            self.skip_enabled = True
        else:
            self.skip_enabled = False

        # projection convolution
        self.project_conv = nn.Conv2d(in_channels=exp_channels,
                                      out_channels=out_channels,
                                      kernel_size=(1, 1),
                                      bias=bias)
        self.project_bn = nn.BatchNorm2d(out_channels, **bn_kwargs)
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x

        if self.expansion_conv_enabled:
            # expansion convolution applied only if expansion ratio > 1
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.activation(x)

        # depth-wise convolution
        x = self.dp_conv(x)
        x = self.dp_bn(x)
        x = self.activation(x)

        # squeeze-and-excitate
        if self.se_enabled:
            x = self.se(x)

        # projection convolution
        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.skip_enabled:
            # drop-connect applied only if skip connection enabled
            if self.drop_connect_enabled:
                x = self.drop_connect(x)
            x = x + inp
        return x


def round_filters(filters, width_coefficient, depth_divisor=8):
    """Round number of filters based on depth multiplier."""
    min_depth = depth_divisor

    filters *= width_coefficient
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    return int(ceil(depth_coefficient * repeats))


class EfficientNet(nn.Module):
    # (width_coefficient, depth_coefficient, dropout_rate, input_spatial_shape)
    coefficients = [
        (1.0, 1.0, 0.2, 224),
        (1.0, 1.1, 0.2, 240),
        (1.1, 1.2, 0.3, 260),
        (1.2, 1.4, 0.3, 300),
        (1.4, 1.8, 0.4, 380),
        (1.6, 2.2, 0.4, 456),
        (1.8, 2.6, 0.5, 528),
        (2.0, 3.1, 0.5, 600),
    ]

    # block_repeat, kernel_size, stride, expansion_factor, input_channels, output_channels, se_ratio
    block_args = [
        [1, 3, 1, 1, 32, 16, 4],
        [2, 3, 2, 6, 16, 24, 4],
        [2, 5, 2, 6, 24, 40, 4],
        [3, 3, 2, 6, 40, 80, 4],
        [3, 5, 1, 6, 80, 112, 4],
        [4, 5, 2, 6, 112, 192, 4],
        [1, 3, 1, 6, 192, 320, 4],
    ]

    state_dict_urls = [
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b0-e6c39902.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b1-dd8fbb83.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b2-193ca240.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b3-f6776323.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b4-4facd840.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b5-f1219ce4.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b6-745f3bbc.pth",
        "https://storage.googleapis.com/abhuse/pretrained_models/efficientnet/efficientnet-b7-460e9c0f.pth",
    ]

    def __init__(self,
                 b,
                 in_channels=3,
                 n_classes=1000,
                 input_spatial_shape=None,
                 activation=Swish(),
                 bias=False,
                 drop_connect_rate=0.2,
                 override_dropout_rate=None,
                 bn_epsilon=1e-3,
                 bn_momentum=0.01,
                 pretrained=False,
                 progress=False):
        """
        Initialize new EfficientNet model
        :param b: model index, i.e. 0 for EfficientNet-B0
        :param in_channels: number of input channels
        :param n_classes: number of output classes
        :param input_spatial_shape: input image shape
        :param activation: activation function
        :param bias: enable bias in convolution operations
        :param drop_connect_rate: DropConnect rate
        :param override_dropout_rate: dropout rate, this will override default rate for each model
        :param bn_epsilon: batch normalization epsilon
        :param bn_momentum: batch normalization momentum
        :param pretrained: initialize model with weights pre-trained on ImageNet
        :param progress: show progress when downloading pre-trained weights
        """

        super(EfficientNet, self).__init__()

        # verify all parameters
        EfficientNet._check_init_params(b,
                                        in_channels,
                                        n_classes,
                                        input_spatial_shape,
                                        activation,
                                        bias,
                                        drop_connect_rate,
                                        override_dropout_rate,
                                        bn_epsilon,
                                        bn_momentum,
                                        pretrained,
                                        progress)

        self.b = b
        self.in_channels = in_channels
        self.activation = F.relu if activation is None else activation

        width_coefficient, depth_coefficient, \
        dropout_rate, spatial_shape = EfficientNet.coefficients[self.b]

        if input_spatial_shape is not None:
            self.input_spatial_shape = _pair(input_spatial_shape)
        else:
            self.input_spatial_shape = _pair(spatial_shape)

        if override_dropout_rate is not None:
            dropout_rate = override_dropout_rate

        # initial convolution
        init_conv_out_channels = round_filters(32, width_coefficient)
        self.init_conv = SamePaddingConv2d(input_spatial_shape=self.input_spatial_shape,
                                           in_channels=self.in_channels,
                                           out_channels=init_conv_out_channels,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           bias=bias)
        self.init_bn = nn.BatchNorm2d(init_conv_out_channels,
                                      eps=bn_epsilon, momentum=bn_momentum)
        spatial_shape = self.init_conv.out_spatial_shape

        self.blocks = nn.ModuleList()

        n_repeats = [round_repeats(arg[0], depth_coefficient) for arg in EfficientNet.block_args]
        n_blocks = sum(n_repeats)

        block_id = 0
        for block_arg, n_repeat in zip(EfficientNet.block_args, n_repeats):
            kernel_size = block_arg[1]
            stride = block_arg[2]
            expansion_factor = block_arg[3]
            block_in_channels = round_filters(block_arg[4], width_coefficient)
            block_out_channels = round_filters(block_arg[5], width_coefficient)

            # repeat blocks
            for k in range(n_repeat):
                if drop_connect_rate is None:
                    block_dc_rate = None
                else:
                    block_dc_rate = drop_connect_rate * block_id / n_blocks

                se_size = max(1, block_in_channels // block_arg[6])
                _block = MBConvBlock(input_spatial_shape=spatial_shape,
                                     in_channels=block_in_channels,
                                     expansion_factor=expansion_factor,
                                     kernel_size=kernel_size,
                                     out_channels=block_out_channels,
                                     stride=stride,
                                     se_size=se_size,
                                     activation=self.activation,
                                     bn_epsilon=bn_epsilon,
                                     bn_momentum=bn_momentum,
                                     drop_connect_rate=block_dc_rate,
                                     bias=bias)
                spatial_shape = _block.out_spatial_shape
                self.blocks.append(_block)

                # remaining MBConv blocks of the group have stride 1 and in_channels=out_channels
                stride = 1
                block_in_channels = block_out_channels
                block_id += 1

        head_conv_out_channels = round_filters(1280, width_coefficient)
        self.head_conv = nn.Conv2d(in_channels=self.blocks[-1].project_conv.out_channels,
                                   out_channels=head_conv_out_channels,
                                   kernel_size=(1, 1),
                                   bias=bias)
        self.head_bn = nn.BatchNorm2d(head_conv_out_channels,
                                      eps=bn_epsilon, momentum=bn_momentum)

        if dropout_rate > 0:
            self.dropout_enabled = True
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_enabled = False

        self.avpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(head_conv_out_channels, n_classes)

        if pretrained:
            self._load_state(self.b, in_channels, n_classes, progress)

    def _load_state(self, b, in_channels, n_classes, progress):
        state_dict = model_zoo.load_url(EfficientNet.state_dict_urls[b], progress=progress)
        strict = True
        if in_channels != 3:
            state_dict.pop('init_conv.weight')
            strict = False
        if n_classes != 1000:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            strict = False
        self.load_state_dict(state_dict, strict=strict)
        print("Model weights loaded successfully.")

    def check_input(self, x):
        if x.dim() != 4:
            raise ValueError("Input x must be 4 dimensional tensor, got {} instead".format(x.dim()))
        if x.size(1) != self.in_channels:
            raise ValueError("Input must have {} channels, got {} instead".format(self.in_channels,
                                                                                  x.size(1)))

    @staticmethod
    def _check_init_params(b,
                           in_channels,
                           n_classes,
                           input_spatial_shape,
                           activation,
                           bias,
                           drop_connect_rate,
                           override_dropout_rate,
                           bn_epsilon,
                           bn_momentum,
                           pretrained,
                           progress):

        if not isinstance(b, int):
            raise ValueError("b must be int, got {} instead".format(type(b)))
        elif not 0 <= b < len(EfficientNet.coefficients):
            raise ValueError("b must be in range 0<=b<=7, got {} instead".format(b))

        if not isinstance(in_channels, int):
            raise ValueError("in_channels must be int, got {} instead".format(type(in_channels)))
        elif not in_channels > 0:
            raise ValueError("in_channels must be > 0, got {} instead".format(in_channels))

        if not isinstance(n_classes, int):
            raise ValueError("n_classes must be int, got {} instead".format(type(n_classes)))
        elif not n_classes > 0:
            raise ValueError("n_classes must be > 0, got {} instead".format(n_classes))

        if not (input_spatial_shape is None or
                isinstance(input_spatial_shape, int) or
                (isinstance(input_spatial_shape, container_abcs.Iterable) and
                 len(input_spatial_shape) == 2 and
                 all(isinstance(s, int) for s in input_spatial_shape))):
            raise ValueError("input_spatial_shape must be either None, int or iterable of ints of length 2"
                             ", got {} instead".format(input_spatial_shape))

        if activation is not None and not callable(activation):
            raise ValueError("activation must be callable but is not")

        if not isinstance(bias, bool):
            raise ValueError("bias must be bool, got {} instead".format(type(bias)))

        if not isinstance(drop_connect_rate, float):
            raise ValueError("drop_connect_rate must be float, got {} instead".format(type(drop_connect_rate)))
        elif not 0 <= drop_connect_rate < 1.0:
            raise ValueError("drop_connect_rate must be within range 0 <= drop_connect_rate < 1.0, "
                             "got {} instead".format(drop_connect_rate))

        if override_dropout_rate is not None:
            if not isinstance(override_dropout_rate, float):
                raise ValueError("override_dropout_rate must be either None or float, "
                                 "got {} instead".format(type(override_dropout_rate)))
            elif not 0 <= override_dropout_rate < 1.0:
                raise ValueError("override_dropout_rate must be within range 0 <= override_dropout_rate < 1.0, "
                                 "got {} instead".format(override_dropout_rate))

        if not isinstance(bn_epsilon, float):
            raise ValueError("bn_epsilon must be float, got {} instead".format(bn_epsilon))

        if not isinstance(bn_momentum, float):
            raise ValueError("bn_momentum must be float, got {} instead".format(bn_momentum))

        if not isinstance(pretrained, bool):
            raise ValueError("pretrained must be bool, got {} instead".format(type(pretrained)))

        if not isinstance(progress, bool):
            raise ValueError("progress must be bool, got {} instead".format(type(progress)))

    def forward(self, x):
        self.check_input(x)

        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.activation(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.activation(x)

        x = self.avpool(x)
        x = torch.flatten(x, 1)

        if self.dropout_enabled:
            x = self.dropout(x)
        x = self.fc(x)

        return x

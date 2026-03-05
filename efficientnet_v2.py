import collections.abc as container_abc
from collections import OrderedDict
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo


def _pair(x):
    if isinstance(x, container_abc.Iterable):
        return x
    return (x, x)


def torch_conv_out_spatial_shape(in_spatial_shape, kernel_size, stride):
    if in_spatial_shape is None:
        return None
    # in_spatial_shape -> [H,W]
    hin, win = _pair(in_spatial_shape)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)

    # dilation and padding are ignored since they are always fixed in efficientnetV2
    hout = int(floor((hin - kh - 1) / sh + 1))
    wout = int(floor((win - kw - 1) / sw + 1))
    return hout, wout


def get_activation(act_fn: str, **kwargs):
    if act_fn in ("silu", "swish"):
        return nn.SiLU(**kwargs)
    elif act_fn == "relu":
        return nn.ReLU(**kwargs)
    elif act_fn == "relu6":
        return nn.ReLU6(**kwargs)
    elif act_fn == "elu":
        return nn.ELU(**kwargs)
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    elif act_fn == "selu":
        return nn.SELU(**kwargs)
    elif act_fn == "mish":
        return nn.Mish(**kwargs)
    else:
        raise ValueError("Unsupported act_fn {}".format(act_fn))


def round_filters(filters, width_coefficient, depth_divisor=8):
    """Round number of filters based on depth multiplier."""
    min_depth = depth_divisor
    filters *= width_coefficient
    new_filters = max(
        min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    return int(ceil(depth_coefficient * repeats))


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
            random_tensor = self.keep_prob + torch.rand(
                [x.size(0), 1, 1, 1], dtype=x.dtype, device=x.device
            )
            binary_tensor = torch.floor(random_tensor)
            return torch.mul(torch.div(x, self.keep_prob), binary_tensor)
        else:
            return x


class SamePaddingConv2d(nn.Module):
    def __init__(
        self,
        in_spatial_shape,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        enforce_in_spatial_shape=False,
        **kwargs,
    ):
        super(SamePaddingConv2d, self).__init__()

        self._in_spatial_shape = _pair(in_spatial_shape)
        # e.g. throw exception if input spatial shape does not match in_spatial_shape
        # when calling self.forward()
        self.enforce_in_spatial_shape = enforce_in_spatial_shape
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)

        in_height, in_width = self._in_spatial_shape
        filter_height, filter_width = kernel_size
        stride_heigth, stride_width = stride
        dilation_height, dilation_width = dilation

        out_height = int(ceil(float(in_height) / float(stride_heigth)))
        out_width = int(ceil(float(in_width) / float(stride_width)))

        pad_along_height = max(
            (out_height - 1) * stride_heigth
            + filter_height
            + (filter_height - 1) * (dilation_height - 1)
            - in_height,
            0,
        )
        pad_along_width = max(
            (out_width - 1) * stride_width
            + filter_width
            + (filter_width - 1) * (dilation_width - 1)
            - in_width,
            0,
        )

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        if any(p > 0 for p in paddings):
            self.zero_pad = nn.ZeroPad2d(paddings)
        else:
            self.zero_pad = None
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

        self._out_spatial_shape = (out_height, out_width)

    @property
    def out_spatial_shape(self):
        return self._out_spatial_shape

    def check_spatial_shape(self, x):
        if (
            x.size(2) != self._in_spatial_shape[0]
            or x.size(3) != self._in_spatial_shape[1]
        ):
            raise ValueError(
                "Expected input spatial shape {}, got {} instead".format(
                    self._in_spatial_shape, x.shape[2:]
                )
            )

    def forward(self, x):
        if self.enforce_in_spatial_shape:
            self.check_spatial_shape(x)
        if self.zero_pad is not None:
            x = self.zero_pad(x)
        x = self.conv(x)
        return x


class SqueezeExcitate(nn.Module):
    def __init__(self, in_channels, se_size, activation=None):
        super(SqueezeExcitate, self).__init__()
        self.dim_reduce = nn.Conv2d(
            in_channels=in_channels, out_channels=se_size, kernel_size=1
        )
        self.dim_restore = nn.Conv2d(
            in_channels=se_size, out_channels=in_channels, kernel_size=1
        )
        self.activation = F.relu if activation is None else activation

    def forward(self, x):
        inp = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dim_reduce(x)
        x = self.activation(x)
        x = self.dim_restore(x)
        x = torch.sigmoid(x)
        return torch.mul(inp, x)


class MBConvBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expansion_factor,
        act_fn,
        act_kwargs=None,
        bn_epsilon=None,
        bn_momentum=None,
        se_size=None,
        drop_connect_rate=None,
        bias=False,
        tf_style_conv=False,
        in_spatial_shape=None,
    ):

        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        exp_channels = in_channels * expansion_factor

        self.ops_lst = []

        # expansion convolution
        if expansion_factor != 1:
            self.expand_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=exp_channels,
                kernel_size=1,
                bias=bias,
            )

            self.expand_bn = nn.BatchNorm2d(
                num_features=exp_channels, eps=bn_epsilon, momentum=bn_momentum
            )

            self.expand_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.extend([self.expand_conv, self.expand_bn, self.expand_act])

        # depth-wise convolution
        if tf_style_conv:
            self.dp_conv = SamePaddingConv2d(
                in_spatial_shape=in_spatial_shape,
                in_channels=exp_channels,
                out_channels=exp_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=exp_channels,
                bias=bias,
            )
            self.out_spatial_shape = self.dp_conv.out_spatial_shape
        else:
            self.dp_conv = nn.Conv2d(
                in_channels=exp_channels,
                out_channels=exp_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                groups=exp_channels,
                bias=bias,
            )
            self.out_spatial_shape = torch_conv_out_spatial_shape(
                in_spatial_shape, kernel_size, stride
            )

        self.dp_bn = nn.BatchNorm2d(
            num_features=exp_channels, eps=bn_epsilon, momentum=bn_momentum
        )

        self.dp_act = get_activation(act_fn, **act_kwargs)
        self.ops_lst.extend([self.dp_conv, self.dp_bn, self.dp_act])

        # Squeeze and Excitate
        if se_size is not None:
            self.se = SqueezeExcitate(
                exp_channels, se_size, activation=get_activation(act_fn, **act_kwargs)
            )
            self.ops_lst.append(self.se)

        # projection layer
        self.project_conv = nn.Conv2d(
            in_channels=exp_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

        self.project_bn = nn.BatchNorm2d(
            num_features=out_channels, eps=bn_epsilon, momentum=bn_momentum
        )

        # no activation function in projection layer

        self.ops_lst.extend([self.project_conv, self.project_bn])

        self.skip_enabled = in_channels == out_channels and stride == 1

        if self.skip_enabled and drop_connect_rate is not None:
            self.drop_connect = DropConnect(drop_connect_rate)
            self.ops_lst.append(self.drop_connect)

    def forward(self, x):
        inp = x
        for op in self.ops_lst:
            x = op(x)
        if self.skip_enabled:
            return x + inp
        else:
            return x


class FusedMBConvBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        expansion_factor,
        act_fn,
        act_kwargs=None,
        bn_epsilon=None,
        bn_momentum=None,
        se_size=None,
        drop_connect_rate=None,
        bias=False,
        tf_style_conv=False,
        in_spatial_shape=None,
    ):

        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        exp_channels = in_channels * expansion_factor

        self.ops_lst = []

        # expansion convolution
        expansion_out_shape = in_spatial_shape
        if expansion_factor != 1:
            if tf_style_conv:
                self.expand_conv = SamePaddingConv2d(
                    in_spatial_shape=in_spatial_shape,
                    in_channels=in_channels,
                    out_channels=exp_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                )
                expansion_out_shape = self.expand_conv.out_spatial_shape
            else:
                self.expand_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=exp_channels,
                    kernel_size=kernel_size,
                    padding=1,
                    stride=stride,
                    bias=bias,
                )
                expansion_out_shape = torch_conv_out_spatial_shape(
                    in_spatial_shape, kernel_size, stride
                )

            self.expand_bn = nn.BatchNorm2d(
                num_features=exp_channels, eps=bn_epsilon, momentum=bn_momentum
            )

            self.expand_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.extend([self.expand_conv, self.expand_bn, self.expand_act])

        # Squeeze and Excitate
        if se_size is not None:
            self.se = SqueezeExcitate(
                exp_channels, se_size, activation=get_activation(act_fn, **act_kwargs)
            )
            self.ops_lst.append(self.se)

        # projection layer
        kernel_size = 1 if expansion_factor != 1 else kernel_size
        stride = 1 if expansion_factor != 1 else stride
        if tf_style_conv:
            self.project_conv = SamePaddingConv2d(
                in_spatial_shape=expansion_out_shape,
                in_channels=exp_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
            self.out_spatial_shape = self.project_conv.out_spatial_shape
        else:
            self.project_conv = nn.Conv2d(
                in_channels=exp_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1 if kernel_size > 1 else 0,
                bias=bias,
            )
            self.out_spatial_shape = torch_conv_out_spatial_shape(
                expansion_out_shape, kernel_size, stride
            )

        self.project_bn = nn.BatchNorm2d(
            num_features=out_channels, eps=bn_epsilon, momentum=bn_momentum
        )

        self.ops_lst.extend([self.project_conv, self.project_bn])

        if expansion_factor == 1:
            self.project_act = get_activation(act_fn, **act_kwargs)
            self.ops_lst.append(self.project_act)

        self.skip_enabled = in_channels == out_channels and stride == 1

        if self.skip_enabled and drop_connect_rate is not None:
            self.drop_connect = DropConnect(drop_connect_rate)
            self.ops_lst.append(self.drop_connect)

    def forward(self, x):
        inp = x
        for op in self.ops_lst:
            x = op(x)
        if self.skip_enabled:
            return x + inp
        else:
            return x


class EfficientNetV2(nn.Module):
    _models = {
        "b0": {
            "num_repeat": [1, 2, 2, 3, 5, 8],
            "kernel_size": [3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2],
            "expand_ratio": [1, 4, 4, 4, 6, 6],
            "in_channel": [32, 16, 32, 48, 96, 112],
            "out_channel": [16, 32, 48, 96, 112, 192],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, True],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "train_size": 192,
            "eval_size": 224,
            "dropout": 0.2,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_b0_21k_ft1k-a91e14c5.pth",
            "model_name": "efficientnet_v2_b0_21k_ft1k-a91e14c5.pth",
        },
        "b1": {
            "num_repeat": [1, 2, 2, 3, 5, 8],
            "kernel_size": [3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2],
            "expand_ratio": [1, 4, 4, 4, 6, 6],
            "in_channel": [32, 16, 32, 48, 96, 112],
            "out_channel": [16, 32, 48, 96, 112, 192],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, True],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "train_size": 192,
            "eval_size": 240,
            "dropout": 0.2,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_b1_21k_ft1k-58f4fb47.pth",
            "model_name": "efficientnet_v2_b1_21k_ft1k-58f4fb47.pth",
        },
        "b2": {
            "num_repeat": [1, 2, 2, 3, 5, 8],
            "kernel_size": [3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2],
            "expand_ratio": [1, 4, 4, 4, 6, 6],
            "in_channel": [32, 16, 32, 48, 96, 112],
            "out_channel": [16, 32, 48, 96, 112, 192],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, True],
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "train_size": 208,
            "eval_size": 260,
            "dropout": 0.3,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_b2_21k_ft1k-db4ac0ee.pth",
            "model_name": "efficientnet_v2_b2_21k_ft1k-db4ac0ee.pth",
        },
        "b3": {
            "num_repeat": [1, 2, 2, 3, 5, 8],
            "kernel_size": [3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2],
            "expand_ratio": [1, 4, 4, 4, 6, 6],
            "in_channel": [32, 16, 32, 48, 96, 112],
            "out_channel": [16, 32, 48, 96, 112, 192],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, True],
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            "train_size": 240,
            "eval_size": 300,
            "dropout": 0.3,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_b3_21k_ft1k-3da5874c.pth",
            "model_name": "efficientnet_v2_b3_21k_ft1k-3da5874c.pth",
        },
        "s": {
            "num_repeat": [2, 4, 4, 6, 9, 15],
            "kernel_size": [3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2],
            "expand_ratio": [1, 4, 4, 4, 6, 6],
            "in_channel": [24, 24, 48, 64, 128, 160],
            "out_channel": [24, 48, 64, 128, 160, 256],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, True],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "train_size": 300,
            "eval_size": 384,
            "dropout": 0.2,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_s_21k_ft1k-dbb43f38.pth",
            "model_name": "efficientnet_v2_s_21k_ft1k-dbb43f38.pth",
        },
        "m": {
            "num_repeat": [3, 5, 5, 7, 14, 18, 5],
            "kernel_size": [3, 3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2, 1],
            "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
            "in_channel": [24, 24, 48, 80, 160, 176, 304],
            "out_channel": [24, 48, 80, 160, 176, 304, 512],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, False, True],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "train_size": 384,
            "eval_size": 480,
            "dropout": 0.3,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_m_21k_ft1k-da8e56c0.pth",
            "model_name": "efficientnet_v2_m_21k_ft1k-da8e56c0.pth",
        },
        "l": {
            "num_repeat": [4, 7, 7, 10, 19, 25, 7],
            "kernel_size": [3, 3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2, 1],
            "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
            "in_channel": [32, 32, 64, 96, 192, 224, 384],
            "out_channel": [32, 64, 96, 192, 224, 384, 640],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, False, True],
            "feature_stages": [1, 2, 4, 6],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "train_size": 384,
            "eval_size": 480,
            "dropout": 0.4,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_l_21k_ft1k-08121eee.pth",
            "model_name": "efficientnet_v2_l_21k_ft1k-08121eee.pth",
        },
        "xl": {
            "num_repeat": [4, 8, 8, 16, 24, 32, 8],
            "kernel_size": [3, 3, 3, 3, 3, 3, 3],
            "stride": [1, 2, 2, 2, 1, 2, 1],
            "expand_ratio": [1, 4, 4, 4, 6, 6, 6],
            "in_channel": [32, 32, 64, 96, 192, 256, 512],
            "out_channel": [32, 64, 96, 192, 256, 512, 640],
            "se_ratio": [None, None, None, 0.25, 0.25, 0.25, 0.25],
            "conv_type": [1, 1, 1, 0, 0, 0, 0],
            "is_feature_stage": [False, True, True, False, True, False, True],
            "feature_stages": [1, 2, 4, 6],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "train_size": 384,
            "eval_size": 512,
            "dropout": 0.4,
            "weight_url": "https://huggingface.co/abhuse/efficientnet/resolve/main/efficientnet_v2_xl_21k_ft1k-1fcc9744.pth",
            "model_name": "efficientnet_v2_xl_21k_ft1k-1fcc9744.pth",
        },
    }

    def __init__(
        self,
        model_name,
        in_channels=3,
        n_classes=1000,
        tf_style_conv=False,
        in_spatial_shape=None,
        activation="silu",
        activation_kwargs=None,
        bias=False,
        drop_connect_rate=0.2,
        dropout_rate=None,
        bn_epsilon=1e-3,
        bn_momentum=0.01,
        pretrained=False,
        progress=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.model_name = model_name
        self.cfg = self._models[model_name]

        if tf_style_conv and in_spatial_shape is None:
            in_spatial_shape = self.cfg["eval_size"]

        activation_kwargs = {} if activation_kwargs is None else activation_kwargs
        dropout_rate = self.cfg["dropout"] if dropout_rate is None else dropout_rate
        _input_ch = in_channels

        self.feature_block_ids = []

        # Precompute all layer configurations
        self.layer_configs = self._precompute_layer_configs(
            in_channels=in_channels,
            n_classes=n_classes,
            tf_style_conv=tf_style_conv,
            activation=activation,
            activation_kwargs=activation_kwargs,
            bias=bias,
            drop_connect_rate=drop_connect_rate,
            dropout_rate=dropout_rate,
            bn_epsilon=bn_epsilon,
            bn_momentum=bn_momentum,
        )

        # stem - using precomputed configuration but calculating spatial shapes dynamically
        stem_config = self.layer_configs["stem"]
        if stem_config["tf_style_conv"]:
            self.stem_conv = SamePaddingConv2d(
                in_spatial_shape=in_spatial_shape,
                in_channels=stem_config["in_channels"],
                out_channels=stem_config["out_channels"],
                kernel_size=stem_config["kernel_size"],
                stride=stem_config["stride"],
                bias=stem_config["bias"],
            )
            in_spatial_shape = self.stem_conv.out_spatial_shape
            # Save calculated spatial shapes back to config
            stem_config["in_spatial_shape"] = in_spatial_shape
            stem_config["out_spatial_shape"] = self.stem_conv.out_spatial_shape
        else:
            self.stem_conv = nn.Conv2d(
                in_channels=stem_config["in_channels"],
                out_channels=stem_config["out_channels"],
                kernel_size=stem_config["kernel_size"],
                stride=stem_config["stride"],
                padding=stem_config["padding"],
                bias=stem_config["bias"],
            )

        self.stem_bn = nn.BatchNorm2d(
            num_features=stem_config["out_channels"],
            eps=stem_config["bn_epsilon"],
            momentum=stem_config["bn_momentum"],
        )

        self.stem_act = get_activation(
            stem_config["activation"],
            **stem_config["activation_kwargs"],
        )

        # stages - using precomputed configurations but calculating spatial shapes dynamically
        for stage_config in self.layer_configs["stages"]:
            for block_config in stage_config["blocks"]:
                conv_block = (
                    MBConvBlockV2
                    if block_config["conv_block_type"] == "MBConvBlockV2"
                    else FusedMBConvBlockV2
                )

                # Save current in_spatial_shape to config before creating block
                block_config["in_spatial_shape"] = in_spatial_shape

                _b = conv_block(
                    in_channels=block_config["in_channels"],
                    out_channels=block_config["out_channels"],
                    kernel_size=block_config["kernel_size"],
                    stride=block_config["stride"],
                    expansion_factor=block_config["expansion_factor"],
                    act_fn=block_config["act_fn"],
                    act_kwargs=block_config["act_kwargs"],
                    bn_epsilon=block_config["bn_epsilon"],
                    bn_momentum=block_config["bn_momentum"],
                    se_size=block_config["se_size"],
                    drop_connect_rate=block_config["drop_connect_rate"],
                    bias=block_config["bias"],
                    tf_style_conv=block_config["tf_style_conv"],
                    in_spatial_shape=in_spatial_shape,
                )
                self.blocks.append(_b)

                if tf_style_conv:
                    block_config["out_spatial_shape"] = _b.out_spatial_shape
                    # in_spatial_shape of next block is the out_spatial_shape of current block
                    in_spatial_shape = _b.out_spatial_shape

            if stage_config["is_feature_stage"]:
                self.feature_block_ids.append(len(self.blocks) - 1)

        # head - using precomputed configuration with spatial shape tracking
        head_config = self.layer_configs["head"]
        # Save spatial shape to config
        head_config["in_spatial_shape"] = in_spatial_shape
        head_config["out_spatial_shape"] = (
            in_spatial_shape  # 1x1 conv doesn't change spatial shape
        )

        self.head_conv = nn.Conv2d(
            in_channels=head_config["in_channels"],
            out_channels=head_config["out_channels"],
            kernel_size=head_config["kernel_size"],
            bias=head_config["bias"],
        )
        self.head_bn = nn.BatchNorm2d(
            num_features=head_config["out_channels"],
            eps=head_config["bn_epsilon"],
            momentum=head_config["bn_momentum"],
        )
        self.head_act = get_activation(
            head_config["activation"],
            **head_config["activation_kwargs"],
        )

        # classifier - using precomputed configuration
        classifier_config = self.layer_configs["classifier"]
        self.dropout = nn.Dropout(p=classifier_config["dropout_rate"])
        self.avpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            classifier_config["in_features"],
            classifier_config["out_features"],
        )

        if pretrained:
            self._load_state(_input_ch, n_classes, progress, tf_style_conv)

        return

    def _precompute_layer_configs(
        self,
        in_channels,
        n_classes,
        tf_style_conv,
        activation,
        activation_kwargs,
        bias,
        drop_connect_rate,
        dropout_rate,
        bn_epsilon,
        bn_momentum,
    ):
        """Precompute all layer configurations for the network (without spatial shapes)."""
        configs = {"stem": {}, "stages": [], "head": {}, "classifier": {}}

        # Calculate stem configuration
        stem_out_channels = self._get_stem_conv_out_channels()
        stem_config = {
            "layer_type": "stem_conv",
            "in_channels": in_channels,
            "out_channels": stem_out_channels,
            "kernel_size": 3,
            "stride": 2,
            "bias": bias,
            "bn_epsilon": bn_epsilon,
            "bn_momentum": bn_momentum,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
            "tf_style_conv": tf_style_conv,
            "padding": 1 if not tf_style_conv else None,
        }

        configs["stem"] = stem_config

        # Calculate stage configurations
        drop_connect_rates = self._get_dropconnect_rates(drop_connect_rate)
        stages = zip(
            *[
                self.cfg[x]
                for x in [
                    "num_repeat",
                    "kernel_size",
                    "stride",
                    "expand_ratio",
                    "se_ratio",
                    "conv_type",
                ]
            ]
        )
        stage_channels = self._get_stage_channels()

        block_idx = 0

        for stage_idx, stage_args in enumerate(stages):
            (num_repeat, kernel_size, stride, expand_ratio, se_ratio, conv_type) = (
                stage_args
            )

            stage_in_channels = stage_channels[stage_idx]["in_channels"]
            stage_out_channels = stage_channels[stage_idx]["out_channels"]
            is_feature_stage = stage_channels[stage_idx]["is_feature_stage"]
            num_repeat = round_repeats(num_repeat, self.cfg["depth_coefficient"])

            conv_block_type = (
                "MBConvBlockV2" if conv_type == 0 else "FusedMBConvBlockV2"
            )

            stage_config = {
                "stage_idx": stage_idx,
                "is_feature_stage": is_feature_stage,
                "conv_block_type": conv_block_type,
                "num_repeat": num_repeat,
                "kernel_size": kernel_size,
                "stride": stride,
                "expand_ratio": expand_ratio,
                "se_ratio": se_ratio,
                "blocks": [],
            }

            # Configure each block in the stage
            current_in_channels = stage_in_channels
            current_stride = stride

            for repeat_idx in range(num_repeat):
                se_size = (
                    None
                    if se_ratio is None
                    else max(1, int(current_in_channels * se_ratio))
                )

                block_config = {
                    "block_idx": block_idx,
                    "stage_idx": stage_idx,
                    "repeat_idx": repeat_idx,
                    "conv_block_type": conv_block_type,
                    "in_channels": current_in_channels,
                    "out_channels": stage_out_channels,
                    "kernel_size": kernel_size,
                    "stride": current_stride,
                    "expansion_factor": expand_ratio,
                    "act_fn": activation,
                    "act_kwargs": activation_kwargs,
                    "bn_epsilon": bn_epsilon,
                    "bn_momentum": bn_momentum,
                    "se_size": se_size,
                    "drop_connect_rate": drop_connect_rates[block_idx],
                    "bias": bias,
                    "tf_style_conv": tf_style_conv,
                    "is_feature_stage": is_feature_stage,
                }

                stage_config["blocks"].append(block_config)

                # Update for next iteration
                current_in_channels = stage_out_channels
                current_stride = 1  # Only first block in stage has stride > 1
                block_idx += 1

            configs["stages"].append(stage_config)

        # Calculate head configuration
        head_conv_out_channels = self._get_head_conv_out_channels()
        head_config = {
            "layer_type": "head_conv",
            "in_channels": stage_out_channels,  # From last stage
            "out_channels": head_conv_out_channels,
            "kernel_size": 1,
            "stride": 1,
            "bias": bias,
            "bn_epsilon": bn_epsilon,
            "bn_momentum": bn_momentum,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
        }
        configs["head"] = head_config

        # Calculate classifier configuration
        classifier_config = {
            "layer_type": "classifier",
            "in_features": head_conv_out_channels,
            "out_features": n_classes,
            "dropout_rate": dropout_rate,
        }
        configs["classifier"] = classifier_config

        return configs

    def _load_state(self, in_channels, n_classes, progress, tf_style_conv):
        state_dict = model_zoo.load_url(
            self.cfg["weight_url"], progress=progress, file_name=self.cfg["model_name"]
        )

        strict = True

        if not tf_style_conv:
            state_dict = OrderedDict(
                [
                    (k.replace(".conv.", "."), v) if ".conv." in k else (k, v)
                    for k, v in state_dict.items()
                ]
            )

        if in_channels != 3:
            if tf_style_conv:
                state_dict.pop("stem_conv.conv.weight")
            else:
                state_dict.pop("stem_conv.weight")
            strict = False

        if n_classes != 1000:
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            strict = False

        self.load_state_dict(state_dict, strict=strict)
        print("Model weights loaded successfully.")

    def _get_dropconnect_rates(self, drop_connect_rate):
        """Get drop connect rates for each block based on the total number of blocks."""
        nr = self.cfg["num_repeat"]
        dc = self.cfg["depth_coefficient"]
        total = sum(round_repeats(nr[i], dc) for i in range(len(nr)))
        return [drop_connect_rate * i / total for i in range(total)]

    def _get_stem_conv_out_channels(self):
        """Get the number of output channels in the stem convolution layer."""
        return round_filters(self.cfg["in_channel"][0], self.cfg["width_coefficient"])

    def _get_head_conv_out_channels(self):
        """Get the number of output channels in the head convolution layer."""
        return round_filters(1280, self.cfg["width_coefficient"])

    def _get_stage_channels(self):
        """Get input and output channels for each stage."""
        stages_info = []
        for i in range(len(self.cfg["in_channel"])):
            in_ch = round_filters(
                self.cfg["in_channel"][i], self.cfg["width_coefficient"]
            )
            out_ch = round_filters(
                self.cfg["out_channel"][i], self.cfg["width_coefficient"]
            )
            stages_info.append(
                {
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "is_feature_stage": self.cfg["is_feature_stage"][i],
                }
            )
        return stages_info

    def get_features(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))

        features = []
        feat_idx = 0
        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx == self.feature_block_ids[feat_idx]:
                features.append(x)
                feat_idx += 1

        return features

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.head_act(self.head_bn(self.head_conv(x)))
        x = self.dropout(torch.flatten(self.avpool(x), 1))
        x = self.fc(x)

        return x

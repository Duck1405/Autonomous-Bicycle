from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True, use_p8=False):
        super().__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8
        self.attention = attention
        self.first_time = first_time

        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2),
            )
            self.p6_to_p7 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))
            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.p6_w1_relu = nn.ReLU()
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1_relu = nn.ReLU()
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2_relu = nn.ReLU()

    def forward(self, inputs):
        if self.attention:
            return self._forward_fast_attention(inputs)
        return self._forward(inputs)

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        weight = self.p6_w1_relu(self.p6_w1)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        weight = self.p5_w1_relu(self.p5_w1)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        weight = self.p4_w1_relu(self.p4_w1)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        weight = self.p3_w1_relu(self.p3_w1)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        weight = self.p4_w2_relu(self.p4_w2)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p4_out = self.conv4_down(self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        weight = self.p5_w2_relu(self.p5_w2)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        weight = self.p6_w2_relu(self.p6_w2)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p6_out = self.conv6_down(self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        weight = self.p7_w2_relu(self.p7_w2)
        weight = weight / (torch.sum(weight, dim=0) + self.epsilon)
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))
        p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))
        p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out


class Conv3x3BNSwish(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.swish = Swish()
        self.upsample = upsample
        self.block = nn.Sequential(
            Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_sp = SeparableConvBlock(out_channels, onnx_export=False)

    def forward(self, x):
        x = self.conv_sp(self.swish(self.block(x)))
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3BNSwish(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3BNSwish(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in {"add", "cat"}:
            raise ValueError(f"Unsupported merge_policy `{policy}`.")
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        return torch.cat(x, dim=1)


class BiFPNDecoder(nn.Module):
    def __init__(self, encoder_depth=5, pyramid_channels=64, segmentation_channels=64, dropout=0.2, merge_policy="add"):
        super().__init__()
        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [5, 4, 3, 2, 1]
            ]
        )
        self.seg_p2 = SegmentationBlock(32, 64, n_upsamples=0)
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, inputs):
        p2, p3, p4, p5, p6, p7 = inputs
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p7, p6, p5, p4, p3])]
        p2 = self.seg_p2(p2)
        p3, p4, p5, p6, p7 = feature_pyramid
        x = self.merge((p2, p3, p4, p5, p6, p7))
        return self.dropout(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        result = tensor * torch.sigmoid(tensor)
        ctx.save_for_backward(tensor)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.saved_variables[0]
        sigmoid_tensor = torch.sigmoid(tensor)
        return grad_output * (sigmoid_tensor * (1 + tensor * (1 - sigmoid_tensor)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        height, width = x.shape[-2:]
        extra_h = (math.ceil(width / self.stride[1]) - 1) * self.stride[1] - width + self.kernel_size[1]
        extra_v = (math.ceil(height / self.stride[0]) - 1) * self.stride[0] - height + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        return self.conv(x)


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        height, width = x.shape[-2:]
        extra_h = (math.ceil(width / self.stride[1]) - 1) * self.stride[1] - width + self.kernel_size[1]
        extra_v = (math.ceil(height / self.stride[0]) - 1) * self.stride[0] - height + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        return self.pool(x)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(f"Unsupported activation `{name}`.")

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from typing import Callable, Optional

class Decoder(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.Conv2dTranspose(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_bn_relu = nn.SequentialCell(
            [nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1, pad_mode='pad'), nn.BatchNorm2d(out_channels), nn.ReLU()])
           
    def construct(self, x1, x2):
        x1 = self.up(x1)
        x = ops.concat((x1, x2), axis=1)
        x = self.conv_bn_relu(x)
        return x 
    
def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, group=groups, has_bias=False, dilation=dilation, pad_mode='pad')

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)

def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

class BasicBlock(nn.Cell):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Cell] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Cell]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        replace_stride_with_dilation = [False, False, False]

        self.layer1 = self._make_layer(block, 64, layer_nums[0])
        self.layer2 = self._make_layer(block, 128, layer_nums[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layer_nums[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layer_nums[3], stride=2, dilate=replace_stride_with_dilation[2])

        channels = [64,64,128,256,512]
        self.decode4 = Decoder(channels[4],channels[3])
        self.decode3 = Decoder(channels[3],channels[2])
        self.decode2 = Decoder(channels[2],channels[1])
        self.decode1 = Decoder(channels[1],channels[0])
        self.decode0_upsample = nn.ResizeBilinear(False)
        decode0_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        decode0_bn = nn.BatchNorm2d(32)
        decode0_relu = nn.ReLU()
        decode0_conv2 = nn.Conv2d(32, num_classes, kernel_size=1,has_bias=False)
        self.decode0 = nn.SequentialCell([decode0_conv1, decode0_bn, decode0_relu, decode0_conv2])

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.SequentialCell(layers)

    def construct(self, x): # pylint: disable=missing-docstring
        encoder = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)

        x = self.layer2(x)
        encoder.append(x)

        x = self.layer3(x)
        encoder.append(x)

        x = self.layer4(x)
        encoder.append(x)

        d4 = self.decode4(encoder[4], encoder[3])
        d3 = self.decode3(d4, encoder[2]) 
        d2 = self.decode2(d3, encoder[1]) 
        d1 = self.decode1(d2, encoder[0])
        d0 = self.decode0_upsample(d1,scale_factor=2)
        out = self.decode0(d0)  

        return out

def resnet34(class_num):
    return ResNet(BasicBlock,
                  [3, 4, 6, 3],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  class_num)

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding(resolution을 유지해주기 위해서)"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  stride , downsample= None) :
        super().__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor)  :
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is True:
            identity = self.downsample(x)

        out += identity ## F(x) + x
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels, out_channels,  stride , downsample= None) :
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, out_channels) ## 줄이고
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride) ## 계산
        self.bn2 = norm_layer(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion) ## 원상 복구
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is True:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block : Type[Union[BasicBlock, Bottleneck]] ,
        layers,
        num_classes: int = 3
        ) :
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer()
        self.relu = nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int]
    ) -> ResNet:
    model = ResNet(block, layers)
    
    return model

def resnet34() -> ResNet:
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3])

def resnet50() -> ResNet:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3])

def resnet152() -> ResNet:
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3])
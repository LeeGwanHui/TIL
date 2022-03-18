import torch
import torch.nn as nn
from torch import Tensor


## 구현을 간편하게 하기 위해서 논문과는 구조가 다르다. 
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, in_channels, out_channels,  stride = 1) :
        super().__init__()
        self.bottleneck = nn.Sequential(
        ## ex in_channels = 64, out_chaannels =64
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), ## 줄이고
        nn.BatchNorm2d(out_channels),nn.ReLU()
        ## 64 -> 64
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), ## 계산
        nn.BatchNorm2d(out_channels),nn.ReLU()
        ## 64 -> 64*4=256
        nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False) ## 원상 복구
        nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU()
        #self.downsample = downsample
        #self.stride = stride

        self.shortcut = nn.Sequential()
        
        ## layer1 말고는 다 처음 block에서 downsampling이 필요하다.
        if stride != 1 or in_channels != out_channels * self.expansion :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.bottleneck(x) + self.shortcut(x)
        x = self.relu(x)

        return x

cfg = [3,4,6,3]

class ResNet_50(nn.Module):
    def __init__(
        self,
        num_classes: int = 3
        ) :
        super().__init__()
        
        self.in_channels = 64
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(self.in_channels),nn.ReLU()
        )
        self.layer1 = self._make_layer(64, cfg[0], stride =1)
        self.layer2 = self._make_layer(128, cfg[1], stride=2)
        self.layer3 = self._make_layer(256, cfg[2], stride=2)
        self.layer4 = self._make_layer(512, cfg[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, num_of_blocks, stride):
        strides = [stride] + [1] * (num_of_blocks - 1) ## ex) layer1의 경우 [1,1,1] ,  layer2의 경우 [2,1,1,1]
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        ## 3 x32 x 32
        x = self.conv_first(x)
        ## 64x16x16
        x = self.layer1(x)
        ## 256x16x16
        x = self.layer2(x)
        ## 512x8x8
        x = self.layer3(x)
        ## 1024x4x4
        x = self.layer4(x)
        ## 2048x2x2
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
"""ResNet-50 S3FD-M backbone producing strides 4–128 (Sec. III-A)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet50Backbone(nn.Module):
    """ResNet-50 trunk + two extra stages for strides 64 and 128.

    Channel layout (Fig. 2): 256, 512, 1024, 2048, 512, 256.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet50(weights=weights)
        # stride-4: after layer1
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # 256, s4
        self.layer2 = net.layer2  # 512, s8
        self.layer3 = net.layer3  # 1024, s16
        self.layer4 = net.layer4  # 2048, s32
        # Extra SSD-style layers for s64 / s128
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        c6 = self.layer6(c5)
        return [c1, c2, c3, c4, c5, c6]

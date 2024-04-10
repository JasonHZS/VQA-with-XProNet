import torch
import torch.nn as nn
from torchvision import models

class ImageFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageFeatureExtractor, self).__init__()
        # 加载预训练的 ResNet
        resnet = models.resnet50(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特征维度
        # 对于 resnet50，它是 2048
        self.out_features = resnet.fc.in_features

    def forward(self, x):
        # 通过 ResNet 提取特征
        x = self.features(x)
        
        # 将特征图展平
        x = x.view(x.size(0), -1)
        return x

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet_features import resnet50_features

def init_conv_weights(layer, weights_std=0.1,  bias=0.02):
    nn.init.normal(layer.weight.data, std=weights_std)
    nn.init.constant(layer.bias.data, val=bias)
    return layer

def conv3x3(in_channels, out_channels, **kwargs):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer

class Idnet(nn.Module):
    def __init__(self):
        super(Idnet, self).__init__()

        self.resnet = resnet50_features(pretrained=True)

        self.conv_layer1 = conv3x3(2048, 1024, stride=2, padding=1)
        self.conv_layer2 = conv3x3(1024, 1024, stride=2, padding=1)
        self.conv_layer3 = conv3x3(1024, 512, stride=2, padding=1)
        self.conv_layer4 = conv3x3(512, 512, stride=1, padding=1)
        self.fc = nn.Linear(512, 3000)
    
    def forward(self, in_layer):
        res2, res3, res4, res5 = self.resnet(in_layer)
        l = F.relu(self.conv_layer1(res5))
        l = F.relu(self.conv_layer2(l))
        l = F.relu(self.conv_layer3(l))
        l = F.relu(self.conv_layer4(l))
        l = l.view(l.size(0), -1)
        l = self.fc(l)

        return l
    
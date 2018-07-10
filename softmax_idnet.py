import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from resnet_features import resnet50_features
from resnet_utilities.layers import conv1x1, conv3x3


class Idnet(nn.Module):
    def __init__(self):
        super(Idnet, self).__init__()

        self.resnet = resnet50_features(pretrained=True)

        self.conv_layer1 = nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1)
        self.conv_layer2 = nn.Conv2d(4096, 2048, kernel_size=3, stride=2, padding=1)
        self.conv_layer3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)
        self.conv_layer4 = nn.Conv2d(1024, 4096, kernel_size=3, stride=1, padding=1)
        #self.fc = nn.Linear(512, 1024)
    
    def forward(self, in_layer):
        res2, res3, res4, res5 = self.resnet(in_layer)
        l = F.relu(self.conv_layer1(res5))
        l = F.relu(self.conv_layer2(l))
        l = F.relu(self.conv_layer3(l))
        l = F.relu(self.conv_layer4(l))
        #l = l.view(l.size(0), -1)
        #l = self.fc(l)

        return l
    
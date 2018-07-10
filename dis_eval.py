from __future__ import print_function

import os
import sys
import argparse
import pdb
import datetime
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss_originalFAN import FocalLoss
from retinanet_originalFAN import RetinaNet
from Idnet import Idnet
from datagen import ListDataset, face_a_ListDataset, face_a_ListDataset_valid
from eval_datagen import face_att_vali_gallery, face_att_vali_probe

from torch.autograd import Variable
from encoder import DataEncoder
import arcface_loss2


def dis_eval():
    pass











































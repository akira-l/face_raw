from __future__ import print_function

import os
import sys
import random
import numpy as np
import csv

import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop



def face_a_dis_valid( root, list_file, train, transform, input_size):
    root = root
    train = train
    transform = transform
    input_size = input_size

    fnames = []
    boxes = []
    ids = []
    encoder = DataEncoder()
    #euclidean_dis = torch.nn.PairwiseDistance()
    #cosine_dis = torch.nn.CosineSimilarity()

    file_list = csv.reader(open(list_file,'r'))
    file_list = list(file_list)
    for csv_content in file_list:
        fnames.append(csv_content[0])
        ids.append(int(csv_content[1]))
    
    valid_name = fnames[-100:]
    valid_ids = ids[-100:]
    valid_pair = []

    valid_sample1 = random.sample(list(zip(valid_name, valid_ids)), 100)
    valid_sample2 = random.sample(list(zip(valid_name[:50], valid_ids[:50])), 50) + \
                            random.sample(list(zip(valid_name[50:], valid_ids[50:])), 50)
    for pair1, pair2 in zip(valid_sample1, valid_sample2):
        pair2_ = [pair2[0], pair2[1]]
        if pair1[0] == pair2[0]:
            if valid_ids.count(pair1[1]) > 1:
                range_count = valid_ids.count(pair1[1])
                tmp_container = valid_name[valid_ids.index(pair1[1]):valid_ids.index(pair1[1])+range_count]
                del(tmp_container[tmp_container.index(pair1[0])])
                pair2_[0] = random.sample(tmp_container, 1)
                pair2_ = tuple([random.sample(tmp_container, 1)[0], pair1[1]])
            else:
                pair2_ = random.sample(list(zip(valid_name, valid_ids)), 1)
                pair2_ = pair2_[0]

        if random.randint(1,3) == 1:
            if valid_ids.count(pair1[1]) > 1:
                range_count = valid_ids.count(pair1[1])
                tmp_container = valid_name[valid_ids.index(pair1[1]):valid_ids.index(pair1[1])+range_count]
                del(tmp_container[tmp_container.index(pair1[0])])
                pair2_ = tuple([random.sample(tmp_container, 1)[0], pair1[1]])
            else:
                pair2_ = random.sample(list(zip(valid_name, valid_ids)), 1)
                pair2_ = pair2_[0]
        valid_pair.append([pair1[0], pair2_[0], float(pair1[1]==pair2_[1]) ] )
        


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])


    print('checkpoint 1')
    trainset = face_a_dis_valid(root="./../face_a/train",
                                  list_file = "./../face_a/train.csv",
                                  train=False, 
                                  transform=transform, 
                                  input_size=224)
    print('checkpoint 2')

    pdb.set_trace()

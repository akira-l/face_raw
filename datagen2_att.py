'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
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
from seg.get_face_seg import seg_attention

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop


########################################################################################################################33
########################################################################################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33


class face_a_ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, transform_att, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.transform_att = transform_att
        self.input_size = input_size
        self.get_att = seg_attention(input_size)
        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1) 

        fnames = []
        self.boxes = []
        ids = []
        self.ids_list = list(range(2874))
        self.im_name_list = []

        self.encoder = DataEncoder()
        
        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)
        # 2874
        for content_counter in range(len(file_list)):
            fnames.append(file_list[content_counter][0])
            ids.append(int(file_list[content_counter][1]))
        
        '''
        for id_counter in range(2874):
            seq_num = ids.index(id_counter)
            self.im_name_list.append(fnames[seq_num])
            del(ids[seq_num])
            del(fnames[seq_num])
        
        self.im_name_valid = fnames[:400]
        self.im_name_train = fnames[400:]+self.im_name_list
        self.ids_valid = ids[:400]
        self.ids_train = ids[400:]+self.ids_list
        '''
        self.im_name_train = fnames
        self.ids_train = ids


    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        size = self.input_size
        fname = self.im_name_train[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path)
        att_map, out_catch = self.get_att.get_att(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        att_map = resize(att_map, boxes, size, test_flag=True)
        img = center_crop(img, boxes, (size,size), test_flag=True)
        att_map = center_crop(att_map, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        att_map = self.transform_att(att_map)
        att_map = torch.floor(100*att_map)
        att_map = self.thresh(att_map)
        
        id_ = self.ids_train[idx]
        
        return img, img_path, id_, att_map

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        img_path = [x[1] for x in batch]
        id_ = [x[2] for x in batch]
        att = [x[3] for x in batch]
        
        img_path_ = []
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        att_inputs = torch.zeros(num_imgs, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            att_inputs[i] = torch.FloatTensor(att[i])
            img_path_.append(img_path[i])
        return inputs, img_path_, id_, att_inputs

    def __len__(self):
        return len(self.ids_train)#len(self.labels)


########################################################################################################################33
########################################################################################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33


class face_a_ListDataset_valid(data.Dataset):
    def __init__(self, root, list_file, train, transform, transform_att, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.transform_att = transform_att
        self.input_size = input_size
        
        self.get_att = seg_attention(input_size)
        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1) 


        fnames = []
        self.boxes = []
        ids = []
        self.ids_list = list(range(2874))
        self.im_name_list = []

        self.encoder = DataEncoder()
        
        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)
        # 2874
        for content_counter in range(len(file_list)):
            fnames.append(file_list[content_counter][0])
            ids.append(int(file_list[content_counter][1]))
        
        for id_counter in range(2874):
            seq_num = ids.index(id_counter)
            self.im_name_list.append(fnames[seq_num])
            del(ids[seq_num])
            del(fnames[seq_num])

        self.im_name_valid = fnames[:400]
        self.im_name_train = fnames[400:]+self.im_name_list
        self.ids_valid = ids[:400]
        self.ids_train = ids[400:]+self.ids_list


    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        size = self.input_size
        fname = self.im_name_valid[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path)
        att_map, out_catch = self.get_att.get_att(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        att_map = resize(att_map, boxes, size, test_flag=True)
        img = center_crop(img, boxes, (size,size), test_flag=True)
        att_map = center_crop(att_map, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        att_map = self.transform_att(att_map)
        att_map = torch.floor(100*att_map)
        att_map = self.thresh(att_map)

        id_ = self.ids_valid[idx]
        
        return img, img_path, id_, att_map

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        img_path = [x[1] for x in batch]
        id_ = [x[2] for x in batch]
        att = [x[3] for x in batch]
        
        img_path_ = []
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        att_inputs = torch.zeros(num_imgs, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            att_inputs[i] = torch.FloatTensor(att[i])
            img_path_.append(img_path[i])
        return inputs, img_path_, id_, att_inputs

    def __len__(self):
        return len(self.ids_valid)#len(self.labels)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
class probe_ListDataset(data.Dataset):
    def __init__(self, root, train, transform, transform_att, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.transform_att = transform_att
        self.input_size = input_size
        self.get_att = seg_attention(input_size)
        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1) 

        fnames = []
        self.boxes = []
        ids = []
        self.ids_list = list(range(2874))
        self.im_name_list = []

        self.encoder = DataEncoder()
        
        file_list = os.listdir(root)
        # 2874
        self.im_name_train = file_list
        self.ids_train = list(range(len(file_list)))



    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        size = self.input_size
        fname = self.im_name_train[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path)
        att_map, out_catch = self.get_att.get_att(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        att_map = resize(att_map, boxes, size, test_flag=True)
        img = center_crop(img, boxes, (size,size), test_flag=True)
        att_map = center_crop(att_map, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        att_map = self.transform_att(att_map)
        att_map = torch.floor(100*att_map)
        att_map = self.thresh(att_map)
        
        id_ = self.ids_train[idx]
        
        return img, img_path, id_, att_map

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        img_path = [x[1] for x in batch]
        id_ = [x[2] for x in batch]
        att = [x[3] for x in batch]
        
        img_path_ = []
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        att_inputs = torch.zeros(num_imgs, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            att_inputs[i] = torch.FloatTensor(att[i])
            img_path_.append(img_path[i])
        return inputs, img_path_, id_, att_inputs

    def __len__(self):
        return len(self.ids_train)#len(self.labels)
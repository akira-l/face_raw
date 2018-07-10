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
########################################################################################################################33
########################################################################################################################33
########################################################################################################################33


class face_a_triplet_att(data.Dataset):
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

        self.fnames = []
        self.boxes = []
        self.ids = []
        self.get_att = seg_attention(input_size)

        self.encoder = DataEncoder()
        
        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)

        for content_counter in range(len(file_list)):
            self.fnames.append(file_list[content_counter][0])
            self.ids.append(int(file_list[content_counter][1]))
            if content_counter > len(file_list)-400:
                break
        
        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1) 



    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''

        ####anchor
        # Load image and boxes.

        triplet_id = []
        triplet_id.append(idx)
        pos_counter = self.ids.count(self.ids(idx))
        pos_id = random.randint(self.ids.count(self.ids(idx)), self.ids.count(self.ids(idx))+pos_counter)
        triplet_id.append(pos_id)
        neg = self.ids(idx)
        neg_id = self.ids[random.randint(0, len(self.ids)-1)]
        while neg == neg_id:
            neg_id = self.ids[random.randint(0, len(self.ids)-1)]
        triplet_id.append(neg_id)

        tri_img = []
        tri_img_path = []
        tri_att_map = []
        for get_idx in triplet_id:
            size = self.input_size
            fname = self.fnames[get_idx]
            img_path = os.path.join(self.root, fname)
            tri_img_path.append(img_path)
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
            tri_img.append(img)
            att_map = self.transform_att(att_map)
            att_map = torch.floor(100*att_map)
            att_map = self.thresh(att_map)
            tri_att_map.append(att_map)
        
        return tri_img, tri_img_path, triplet_id, tri_att_map

    def collate_fn(self, batch):
        '''Pad images and encode targets.
        As for images are of different sizes, we need to pad them to the same size.
        Args:
          batch: (list) of images, cls_targets, loc_targets.
        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        anchor_imgs, pos_imgs, neg_imgs = [x[0] for x in batch]
        anchor_img_path, pos_img_path, neg_img_path = [x[1] for x in batch]
        anchor_id_, pos_id_, neg_id_ = [x[2] for x in batch]
        anchor_att, pos_att, neg_att = [x[3] for x in batch]
        
        anchor_img_path_ = []
        pos_img_path_ = []
        neg_img_path_ = []
        h = w = self.input_size
        num_imgs = len(anchor_id_)

        anchor_inputs = torch.zeros(num_imgs, 3, h, w)
        pos_inputs = torch.zeros(num_imgs, 3, h, w)
        neg_inputs = torch.zeros(num_imgs, 3, h, w)

        anchor_att_in = torch.zeros(num_imgs, h, w)
        pos_att_in = torch.zeros(num_imgs, h, w)
        neg_att_in = torch.zeros(num_imgs, h, w)


        for i in range(num_imgs):
            
            anchor_inputs[i] = anchor_imgs[i]
            pos_inputs[i] = pos_imgs[i]
            neg_inputs[i] = neg_imgs[i]

            anchor_att_in[i] = torch.FloatTensor(anchor_att[i])
            pos_att_in[i] = torch.FloatTensor(pos_att[i])
            neg_att_in[i] = torch.FloatTensor(neg_att[i])

            anchor_img_path_.append(anchor_img_path[i])
            pos_img_path_.append(pos_img_path[i])
            neg_img_path_.append(neg_img_path[i])

        return [anchor_inputs, pos_inputs, neg_inputs], 
               [anchor_img_path_, pos_img_path_, neg_img_path_], 
               [anchor_id_, pos_id_, neg_id_], 
               [anchor_att_in, pos_att_in, neg_att_in]
               


    def __len__(self):
        return len(self.fnames)#len(self.labels)

########################################################################################################################33
########################################################################################################################33
########################################################################################################################33
########################################################################################################################33
########################################################################################################################33



class face_a_triplet_att_valid(data.Dataset):
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

        self.fnames = []
        self.boxes = []
        self.ids = []
        self.get_att = seg_attention(input_size)

        self.encoder = DataEncoder()
        
        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)

        for content_counter in range(len(file_list)):
            self.fnames.append(file_list[len(file_list)-1 - content_counter][0])
            self.ids.append(int(file_list[len(file_list)-1 - content_counter][1]))
            if content_counter == 399:
                break
        
        self.thresh = torch.nn.Hardtanh(min_val=0, max_val=1) 



    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''

        ####anchor
        # Load image and boxes.

        triplet_id = []
        triplet_id.append(idx)
        pos_counter = self.ids.count(self.ids(idx))
        pos_id = random.randint(self.ids.count(self.ids(idx)), self.ids.count(self.ids(idx))+pos_counter)
        triplet_id.append(pos_id)
        neg = self.ids(idx)
        neg_id = self.ids[random.randint(0, len(self.ids)-1)]
        while neg == neg_id:
            neg_id = self.ids[random.randint(0, len(self.ids)-1)]
        triplet_id.append(neg_id)

        tri_img = []
        tri_img_path = []
        tri_att_map = []
        for get_idx in triplet_id:
            size = self.input_size
            fname = self.fnames[get_idx]
            img_path = os.path.join(self.root, fname)
            tri_img_path.append(img_path)
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
            tri_img.append(img)
            att_map = self.transform_att(att_map)
            att_map = torch.floor(100*att_map)
            att_map = self.thresh(att_map)
            tri_att_map.append(att_map)
        
        return tri_img, tri_img_path, triplet_id, tri_att_map

    def collate_fn(self, batch):
        '''Pad images and encode targets.
        As for images are of different sizes, we need to pad them to the same size.
        Args:
          batch: (list) of images, cls_targets, loc_targets.
        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        anchor_imgs, pos_imgs, neg_imgs = [x[0] for x in batch]
        anchor_img_path, pos_img_path, neg_img_path = [x[1] for x in batch]
        anchor_id_, pos_id_, neg_id_ = [x[2] for x in batch]
        anchor_att, pos_att, neg_att = [x[3] for x in batch]
        
        anchor_img_path_ = []
        pos_img_path_ = []
        neg_img_path_ = []
        h = w = self.input_size
        num_imgs = len(anchor_id_)

        anchor_inputs = torch.zeros(num_imgs, 3, h, w)
        pos_inputs = torch.zeros(num_imgs, 3, h, w)
        neg_inputs = torch.zeros(num_imgs, 3, h, w)

        anchor_att_in = torch.zeros(num_imgs, h, w)
        pos_att_in = torch.zeros(num_imgs, h, w)
        neg_att_in = torch.zeros(num_imgs, h, w)


        for i in range(num_imgs):
            
            anchor_inputs[i] = anchor_imgs[i]
            pos_inputs[i] = pos_imgs[i]
            neg_inputs[i] = neg_imgs[i]

            anchor_att_in[i] = torch.FloatTensor(anchor_att[i])
            pos_att_in[i] = torch.FloatTensor(pos_att[i])
            neg_att_in[i] = torch.FloatTensor(neg_att[i])

            anchor_img_path_.append(anchor_img_path[i])
            pos_img_path_.append(pos_img_path[i])
            neg_img_path_.append(neg_img_path[i])

        return [anchor_inputs, pos_inputs, neg_inputs], 
               [anchor_id_, pos_id_, neg_id_], 
               [anchor_img_path_, pos_img_path_, neg_img_path_], 
               [anchor_att_in, pos_att_in, neg_att_in]


    def __len__(self):
        return len(self.fnames)#len(self.labels)
































def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    transform_att = transforms.Compose([
        transforms.ToTensor()
    ])
    '''
    trainset = ListDataset(root="./../../wider face/WIDER_val/images",
                          list_file="./../../wider face/wider_face_split/wider_face_val_bbx_gt.txt",
                          train=True, 
                          transform=transform, 
                          input_size=100)
    '''
    print('checkpoint 1')
    trainset = face_a_ListDataset_with_att(root="./../face_a/train",
                                  list_file = "./../face_a/train.csv",
                                  train=False, 
                                  transform=transform, 
                                  transform_att=transform_att,
                                  input_size=224)
    print('checkpoint 2')

    trainloader = torch.utils.data.DataLoader(trainset, 
                                             batch_size=4, 
                                             shuffle=False, 
                                             num_workers=0, 
                                             collate_fn=trainset.collate_fn)
    print('\n checkpoint 3\n')
    
    for batch_idx, (inputs, img_path, ids, att) in enumerate(trainloader):
        print(inputs.size())
        print(img_path)
        print(ids)
        grid = torchvision.utils.make_grid(inputs, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        pdb.set_trace()
        grid_att = torchvision.utils.make_grid(att, 1)
        torchvision.utils.save_image(grid_att, 'att.png')
        #break
        pdb.set_trace()
    '''
    for batch_idx, (inputs, loc_targets, cls_targets, att_gt, img_path) in enumerate(trainloader):
        print(inputs.size())
        grid = torchvision.utils.make_grid(att_gt, 1)
        torchvision.utils.save_image(grid, 'att.jpg')
        gridi = torchvision.utils.make_grid(inputs, 1)
        torchvision.utils.save_image(gridi, 'img.jpg')
    '''

if __name__ == "__main__":
    test()

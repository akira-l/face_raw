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



class face_a_dis_valid(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.ids = []
        self.encoder = DataEncoder()
        #euclidean_dis = torch.nn.PairwiseDistance()
        #cosine_dis = torch.nn.CosineSimilarity()

        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)
        for csv_content in file_list:
            self.fnames.append(csv_content[0])
            self.ids.append(int(csv_content[1]))
        
        self.valid_name = self.fnames[-400:]
        self.valid_ids = self.ids[-400:]
        self.valid_pair = []

        self.valid_sample1 = random.sample(list(zip(self.valid_name, self.valid_ids)), 400)
        self.valid_sample2 = random.sample(list(zip(self.valid_name[:200], self.valid_ids[:200])), 200) + \
                             random.sample(list(zip(self.valid_name[200:], self.valid_ids[200:])), 200)


        for pair1, pair2 in zip(self.valid_sample1, self.valid_sample2):
            pair2_ = [pair2[0], pair2[1]]
            if pair1[0] == pair2[0]:
                if self.valid_ids.count(pair1[1]) > 1:
                    range_count = self.valid_ids.count(pair1[1])
                    tmp_container = self.valid_name[self.valid_ids.index(pair1[1]):self.valid_ids.index(pair1[1])+range_count]
                    del(tmp_container[tmp_container.index(pair1[0])])
                    pair2_[0] = random.sample(tmp_container, 1)
                    pair2_ = tuple([random.sample(tmp_container, 1)[0], pair1[1]])
                else:
                    pair2_ = random.sample(list(zip(self.valid_name, self.valid_ids)), 1)
                    pair2_ = pair2_[0]

            if random.randint(1,3) == 1:
                if self.valid_ids.count(pair1[1]) > 1:
                    range_count = self.valid_ids.count(pair1[1])
                    tmp_container = self.valid_name[self.valid_ids.index(pair1[1]):self.valid_ids.index(pair1[1])+range_count]
                    del(tmp_container[tmp_container.index(pair1[0])])
                    pair2_ = tuple([random.sample(tmp_container, 1)[0], pair1[1]])
                else:
                    pair2_ = random.sample(list(zip(self.valid_name, self.valid_ids)), 1)
                    pair2_ = pair2_[0]
            self.valid_pair.append([pair1[0], pair2_[0], float(pair1[1]==pair2_[1]) ] )
        

    def __getitem__(self, idx):
        # Load image and boxes.
        size = self.input_size
        fname_pair1 = self.valid_pair[idx][0]
        fname_pair2 = self.valid_pair[idx][1]

        img_path_pair1 = os.path.join(self.root, fname_pair1)
        img_path_pair2 = os.path.join(self.root, fname_pair2)

        img_pair1 = Image.open(img_path_pair1)
        img_pair2 = Image.open(img_path_pair2)

        att_map_pair1, _ = self.get_att.get_att(img_pair1)
        att_map_pair2, _ = self.get_att.get_att(img_pair2)


        if img_pair1.mode != 'RGB':
            img_pair1 = img_pair1.convert('RGB')
        boxes_pair1 = torch.zeros(2,4)
        img_pair1 = resize(img_pair1, boxes_pair1, size, test_flag=True)
        att_map_pair1 = resize(att_map_pair1, img_pair1, size, test_flag=True)

        img_pair1 = self.transform(img_pair1)
        att_map_pair1 = self.transform(att_map_pair1)
        att_map_pair1 = torch.floor(100*att_map_pair1)
        att_map_pair1 = self.thresh(att_map_pair1)

        if img_pair2.mode != 'RGB':
            img_pair2 = img_pair2.convert('RGB')
        boxes_pair2 = torch.zeros(2,4)
        img_pair2 = resize(img_pair2, boxes_pair2, size, test_flag=True)
        att_map_pair2 = resize(att_map_pair2, boxes_pair2, size, test_flag=True)

        img_pair2 = self.transform(img_pair2)
        att_map_pair2 = self.transform(att_map_pair2)
        att_map_pair2 = torch.floor(100*att_map_pair2)
        att_map_pair2 = self.thresh(att_map_pair2)

        return img_pair1, att_map_pair1, img_pair2, att_map_pair2, self.valid_pair[idx][2] 

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        valid_img1 = [x[0] for x in batch]
        valid_att1 = [x[1] for x in batch]
        valid_img2 = [x[2] for x in batch]
        valid_att2 = [x[3] for x in batch]
        valid_flag = [x[4] for x in batch]
        
        h = w = self.input_size
        num_imgs = len(valid_flag)
        valid_input1 = torch.zeros(num_imgs, 3, h, w)
        valid_input2 = torch.zeros(num_imgs, 3, h, w)
        valid_att_input1 = torch.zeros(num_imgs, h, w)
        valid_att_input2 = torch.zeros(num_imgs, h, w)

        for i in range(num_imgs):
            valid_input1[i] = valid_img1[i]
            valid_input2[i] = valid_img2[i]
            valid_att_input1[i] = valid_att1[i]
            valid_att_input2[i] = valid_att2[i]
        return valid_input1, valid_att_input1, valid_input2, valid_att_input2, valid_flag

    def __len__(self):
        return len(self.valid_pair)#len(self.labels)




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
    (self, root, list_file, train, transform, input_size)
    '''
    print('checkpoint 1')
    trainset = face_a_dis_valid(root="./../face_a/train",
                                  list_file = "./../face_a/train.csv",
                                  train=False, 
                                  transform=transform, 
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

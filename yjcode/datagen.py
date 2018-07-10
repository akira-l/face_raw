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



class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
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
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        tag_ = root.split('/')[-2]
        if tag_ == 'WIDER_train':
            list_path_ = "./../../wider face/wider_face_split/shape_train.npy"
            self.shape_list = np.load(list_path_)
        if tag_ == 'WIDER_val':
            list_path_ = "./../../wider face/wider_face_split/shape_val.npy"
            self.shape_list = np.load(list_path_)


        self.encoder = DataEncoder()
        
        line_number = 0

        with open(list_file) as f:
            lines = f.readlines()
            #self.num_samples = len(lines)
            
        while line_number < len(lines)-1:
            fig_name = lines[line_number].split()[0]
            box = []
            label = []
            
            line_number += 1
            face_count = int(lines[line_number])
            line_number += 1
            reserve_counter = 0
            for traverse_face in range(face_count):
                face_data = lines[line_number+traverse_face].split()
                xmin = float(face_data[0])
                ymin = float(face_data[1])
                width = float(face_data[2])
                height = float(face_data[3])
                if width > 30 and height > 30:
                    xmax = xmin+width
                    ymax = ymin+height
                    reserve_counter += 1
                    box.append([xmin,ymin,xmax,ymax])
                    label.append(0.0)
                else:
                    continue
            
            if reserve_counter > 0:
                self.fnames.append(lines[line_number-2].split()[0])
                
                if fig_name[-3:] != 'jpg':
                    raise ValueError("image path error")
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.Tensor(label))
            line_number += face_count
        

            
            
        '''
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        '''
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
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size


        src_shape = self.shape_list[idx]

        att_map = np.zeros([src_shape[0], src_shape[1]])
        
        for att_box in boxes:
            att_map[int(att_box[0]):int(att_box[2]), int(att_box[1]):int(att_box[3])] = 1

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))
        att_map = Image.fromarray(att_map)
        att_map = att_map.resize((size//2, size//2), Image.BILINEAR)

        #img.save('test_in_datagen.jpg')

        img = self.transform(img)
        att_map = self.transform(att_map)
        
        return img, boxes, labels, att_map, img_path

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        att_map = [x[3] for x in batch]
        img_path = [x[4] for x in batch]

        
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        att_gt = torch.zeros(num_imgs, 1, h//2, w//2)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            att_gt[i] = att_map[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), att_gt, img_path
    
    def __len__(self):
        return len(self.labels)
    

########################################################################################################################33
########################################################################################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33
########################################################################################################################33############################################################33


class face_a_ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
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
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.ids = []

        self.encoder = DataEncoder()
        
        file_list = csv.reader(open(list_file,'r'))
        file_list = list(file_list)
        for csv_content in file_list:
            self.fnames.append(csv_content[0])
            self.ids.append(int(csv_content[1]))

        # seg_attention 



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
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        img = center_crop(img, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        id_ = self.ids[idx]
        
        return img, img_path, id_

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
        
        img_path_ = []
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            img_path_.append(img_path[i])
        return inputs, img_path_, id_

    def __len__(self):
        return len(self.fnames)#len(self.labels)


########################################################################################################################33
########################################################################################################################33
########################################################################################################################33
########################################################################################################################33
########################################################################################################################33


class face_a_ListDataset_with_att(data.Dataset):
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
        for csv_content in file_list:
            self.fnames.append(csv_content[0])
            self.ids.append(int(csv_content[1]))
        
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
        # Load image and boxes.
        size = self.input_size
        fname = self.fnames[idx]
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
        #att_map = np.array(att_map, dtype=np.float32)

        id_ = self.ids[idx]
        
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

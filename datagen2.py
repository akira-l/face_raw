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
import cv2
import re
import dlib

import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from seg.get_face_seg import seg_attention

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop




#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################


class face_a_ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size, align=False, addition=False):
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
        self.align = align

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
            fnames.append(os.path.join(self.root, file_list[content_counter][0]))
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
        
        self.ids_train = self.ids_train
        self.im_name_train = self.im_name_train

        if self.align:
            self.detector = dlib.get_frontal_face_detector()
            predicter_path = "./model/shape_predictor_5_face_landmarks.dat"
            self.sp = dlib.shape_predictor(predicter_path)

        if addition:
            addition_dir = '/home/liang/face/face_a/'
            add_csv = list(csv.reader(open('./../face_a/lfwTrain.csv', 'r')))
            add_id = []
            for img_list in add_csv:
                self.ids_train.append(int(img_list[1]))
                folder_dir = re.sub(r'_\d\d\d\d.jpg', '', img_list[0])
                img_dir = os.path.join(os.path.join(addition_dir+'lfw_masked', folder_dir), img_list[0])
                self.im_name_train.append(img_dir)


    def check_list(self):
        return self.im_name_train, self.ids_train


    def alignment(self, img):
        img_cv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        dets = self.detector(img_cv, 1)
        if len(dets)!=0:
            faces = dlib.full_object_detections()
            for det in dets:
                faces.append(self.sp(img_cv, det))
            face_img = dlib.get_face_chips(img_cv, faces, size=160)
            return Image.fromarray(cv2.cvtColor(face_img[0],cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))



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
        img_path = self.im_name_train[idx]
        img = Image.open(img_path)
        if self.align:
            img = self.alignment(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        #img = center_crop(img, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        id_ = self.ids_train[idx]
        
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
        return len(self.im_name_train)#len(self.labels)


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################


class face_a_ListDataset_valid(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size, align=False, addition=False):
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
        self.align = align

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
            fnames.append(os.path.join(self.root, file_list[content_counter][0]))
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

        if self.align:
            self.detector = dlib.get_frontal_face_detector()
            predicter_path = "./model/shape_predictor_5_face_landmarks.dat"
            self.sp = dlib.shape_predictor(predicter_path)

        if addition:
            addition_dir = '/home/liang/face/face_a/'
            add_csv = list(csv.reader(open('./../face_a/lfwTrain.csv', 'r')))
            add_id = []
            for img_list in add_csv:
                self.ids_train.append(int(img_list[1]))
                folder_dir = re.sub(r'_\d\d\d\d.jpg', '', img_list[0])
                img_dir = os.path.join(os.path.join(addition_dir, folder_dir), img_list[0])
                self.im_name_train.append(img_dir)


    def check_list(self):
        return self.im_name_train, self.ids_train


            

    def alignment(self, img):
        img_cv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        dets = self.detector(img_cv, 1)
        if len(dets)!=0:
            faces = dlib.full_object_detections()
            for det in dets:
                faces.append(self.sp(img_cv, det))
            face_img = dlib.get_face_chips(img_cv, faces, size=160)
            return Image.fromarray(cv2.cvtColor(face_img[0],cv2.COLOR_BGR2RGB))
        else:
            return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))


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
        img_path = self.im_name_valid[idx]
        img = Image.open(img_path)
        if self.align:
            img = self.alignment(img)

        if img.mode != 'RGB':
            img = img.convert('RGB')
        boxes = torch.zeros(2,4)
        img = resize(img, boxes, size, test_flag=True)
        #img = center_crop(img, boxes, (size,size), test_flag=True)
        img = self.transform(img)
        id_ = self.ids_valid[idx]
        
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
        return len(self.ids_valid)#len(self.labels)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    trainset = face_a_ListDataset(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224,
                      align=True, 
                      addition=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=4, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset.collate_fn)

    name_list, id_list = trainset.check_list()
    pdb.set_trace()
    for batch_idx, (inputs, img_path, ids) in enumerate(trainloader):
        pdb.set_trace()



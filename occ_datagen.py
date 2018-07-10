'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random
import numpy as np

import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable


from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop
from mask.mask_paste import masked_att


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

        att_map = np.zeros([size, size])

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))
        #img.save('test_in_datagen.jpg')
        for att_box in boxes:
            att_map[int(att_box[1]):int(att_box[3]), int(att_box[0]):int(att_box[2])] = 1
        att_map, img = masked_att(img, boxes, att_map, 2, './mask')

        att_map = Image.fromarray(att_map)
        att_map = att_map.resize((size//2, size//2), Image.BILINEAR)

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
    


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    print('section1')
    dataset = ListDataset(root="./../../wider face/WIDER_val/images",
                          list_file="./../../wider face/wider_face_split/wider_face_val_bbx_gt.txt",
                          train=True, 
                          transform=transform, 
                          input_size=224)
    print('section2')
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=4, 
                                             shuffle=True, 
                                             num_workers=1, 
                                             collate_fn=dataset.collate_fn)
    print('section3')
    for images, loc_targets, cls_targets, att, img_p in dataloader:
        print('section4')
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'img.jpg')
        grid2 = torchvision.utils.make_grid(att, 1)
        torchvision.utils.save_image(grid2, 'att.jpg')
        pdb.set_trace()

if __name__ == "__main__":
    test()
    print('end here')

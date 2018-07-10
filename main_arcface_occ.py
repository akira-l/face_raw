from __future__ import print_function

import os
import sys
import argparse
import pdb
import datetime
import cv2

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
from datagen2_att import *

from torch.autograd import Variable
from encoder import DataEncoder
import arcface_loss2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='softmax for training')
parser.add_argument('--state', '-s', help='training or test stage')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
global tensorboard_counter
tensorboard_counter = 0

batch_size=4


# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_att = transforms.Compose([
    transforms.ToTensor()
])

trainset = face_a_ListDataset(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      transform_att=transform_att,
                      input_size=224)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset.collate_fn)

gallery_list = face_a_ListDataset(root="./../face_a/test_a/gallery",
                      list_file = "./../face_a/test_a_gallery.csv",
                      train=False, 
                      transform=transform, 
                      transform_att=transform_att,
                      input_size=224)
gallery_loader = torch.utils.data.DataLoader(gallery_list, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=gallery_list.collate_fn)

probe_list = probe_ListDataset(root="./../face_a/test_a/probe",
                      train=False, 
                      transform=transform, 
                      transform_att=transform_att,
                      input_size=224)
probe_loader = torch.utils.data.DataLoader(probe_list, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=probe_list.collate_fn)


trainset_valid = face_a_ListDataset_valid(root="./../face_a/train", 
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform,
                      transform_att=transform_att,
                      input_size=224)
trainloader_valid = torch.utils.data.DataLoader(trainset_valid, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset_valid.collate_fn)


net = RetinaNet()
net = torch.nn.DataParallel(net, device_ids=[0])
net.cuda()

id_net = Idnet()
id_net = torch.nn.DataParallel(id_net, device_ids=[0])
id_net.cuda()

#MCP = arcface_loss2.Arcface(1024, 3000).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD([{'params': id_net.parameters()}],#, {'params':MCP.parameters()}], 
                      lr=1e-3, 
                      momentum=0.9, 
                      weight_decay=1e-4)

net.load_state_dict(torch.load("./trained model/originalFAN_model.pth"))
net.eval()
coder = DataEncoder()


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def train(epoch, file_obj, acc):
    print("\n Epoch : %d" % epoch)
    id_net.train()
    for batch_idx, (inputs, img_path, ids, att) in enumerate(trainloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        with torch.no_grad():
            loc_preds, cls_preds = net(inputs)
        boxes = []
        for box_counter in range(inputs.size(0)):
            box, label, score = coder.decode(loc_preds[box_counter].data.cpu(), 
                                             cls_preds[box_counter].data.cpu(), 
                                             (224, 224))
            if box.size(0) == 1:
                boxes.append([float(x) for x in box[0]])
                continue
            tmp_box = [0, 0, 0, 0]
            for box_loop in box: ###shape should be 224!!!!!
                select_box = [float(x) for x in box_loop]
                cond1 = abs(select_box[0]+select_box[2]/2-112)<abs(tmp_box[0]+tmp_box[2]/2-112)
                cond2 = abs(select_box[1]+select_box[3]/2-112)<abs(tmp_box[1]+tmp_box[3]/2-112)
                if cond1 and cond2:
                    tmp_box = select_box
            boxes.append(tmp_box)
        
        img_input = torch.zeros(inputs.size(0), 3, 150, 150)
        for img_counter in range(inputs.size(0)):
            face_box = boxes[img_counter]
            face_box = [int(x) for x in face_box]
            face_box[0] = max(face_box[0], 0)
            face_box[1] = max(face_box[1], 0)
            face_box[2] = min(face_box[2], inputs.size(2))
            face_box[3] = min(face_box[3], inputs.size(2))

            height = face_box[3]-face_box[1]
            width = face_box[2]-face_box[0]
            sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), 
                                 size=(112, 112), 
                                 mode='bilinear')
            att_sampled = F.upsample(att[img_counter, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,1,width,height), 
                                 size=(112, 112), 
                                 mode='bilinear')
            grid = torchvision.utils.make_grid(sampled*att_sampled.cuda())
            torchvision.utils.save_image(grid, './train/'+img_path[img_counter].split('/')[-1])
            print(img_counter)
            #img_input[img_counter, :,:,:] = sampled*att_sampled.cuda()
    return 0
    while 0:
        inputs = Variable(img_input.cuda())
        optimizer.zero_grad()
        id_net_out = id_net(inputs)

        id_out_shape = id_net_out.size()
        target = torch.tensor(ids).view(id_out_shape[0]).cuda()
        #output = MCP(id_net_out, target)
        
        loss = criterion(id_net_out, target)
        loss.backward()
        optimizer.step()

        file_obj.write('epoch|'+str(epoch)+'|loss|'+str(loss)+''+'\n')
        print('epoch: %d | train_loss: %.3f |' % (epoch, loss))
        
        ################################################
        ################################################
        ################################################

        
        if batch_idx%20 == 0:
            estimate = 0
            for batch_idx, (inputs, img_path, ids, att) in enumerate(trainloader_valid):
                inputs = Variable(inputs.cuda(), volatile=True)
                with torch.no_grad():
                    loc_preds, cls_preds = net(inputs)
                boxes = []
                for box_counter in range(inputs.size(0)):
                    #try:
                    box, label, score = coder.decode(loc_preds[box_counter].data.cpu(), 
                                                    cls_preds[box_counter].data.cpu(), 
                                                    (224, 224))

                    if box.size(0) == 1:
                        boxes.append([float(x) for x in box[0]])
                        continue
                    tmp_box = [0, 0, 0, 0]
                    for box_loop in box: ###shape should be 224!!!!!
                        select_box = [float(x) for x in box_loop]
                        cond1 = abs(select_box[0]+select_box[2]/2-112)<abs(tmp_box[0]+tmp_box[2]/2-112)
                        cond2 = abs(select_box[1]+select_box[3]/2-112)<abs(tmp_box[1]+tmp_box[3]/2-112)
                        if cond1 and cond2:
                            tmp_box = select_box
                    boxes.append(tmp_box)
                
                img_input = torch.zeros(inputs.size(0), 3, 150, 150)
                for img_counter in range(inputs.size(0)):
                    face_box = boxes[img_counter]
                    face_box = [int(x) for x in face_box]
                    face_box[0] = max(face_box[0], 0)
                    face_box[1] = max(face_box[1], 0)
                    face_box[2] = min(face_box[2], inputs.size(2))
                    face_box[3] = min(face_box[3], inputs.size(2))

                    height = face_box[3]-face_box[1]
                    width = face_box[2]-face_box[0]
                    sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), 
                                        size=(112, 112), 
                                        mode='bilinear')

                    att_sampled = F.upsample(att[img_counter, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,1,width,height), 
                                        size=(112, 112), 
                                        mode='bilinear')
                    
                    #img_input[img_counter, :,:,:] = sampled*att_sampled.cuda()

                inputs = Variable(img_input.cuda())
                with torch.no_grad():
                    id_net_out = id_net(inputs)
                _, estimate_id = torch.max(id_net_out, dim=1)
                estimate += sum(torch.eq(estimate_id, torch.tensor(ids).cuda()))
            acc_tmp = float(estimate)/100
            print('----------acc:',acc_tmp)
            if acc_tmp > acc and acc>0:
                acc = acc_tmp
                save_model(id_net, 'arcface_id_net_occ-softmax3000-acc'+str(acc)+'.pth')
        
    return acc

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################

def test():
    pass


if __name__ == '__main__':
    acc = 0
    if args.state == 'train':
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if not os.path.exists('./save_file'):
            os.mkdir('./save_file')
        save_file = './save_file/save_file-'+date_now+'.txt'

        file_obj = open(save_file, 'a')
        file_obj.write('description: arcface, with occlusion, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 100, record loss')
        for epoch in range(start_epoch, start_epoch+1):
            acc = train(epoch, file_obj, acc)  
            #test(epoch)
        file_obj.close() 
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')
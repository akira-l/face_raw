from __future__ import print_function

import os
import sys
import argparse
import pdb
import datetime
import cv2
import numpy as np
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss_originalFAN import FocalLoss
from retinanet_originalFAN import RetinaNet
from Idnet2 import Idnet
from datagen2 import face_a_ListDataset, face_a_ListDataset_valid

from torch.autograd import Variable
from encoder import DataEncoder
import arcface_loss2
from cosface_loss import MarginCosineProduct



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='softmax for training')
parser.add_argument('--state', '-s', help='training or test stage')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
global tensorboard_counter
tensorboard_counter = 0

batch_size=72


# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_att = transforms.Compose([
        transforms.ToTensor()
])


trainset = face_a_ListDataset(root="./../face_a/face_image/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224,
                      align=True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset.collate_fn)



trainset_valid = face_a_ListDataset_valid(root="./../face_a/face_image/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224,
                      align=True)
trainloader_valid = torch.utils.data.DataLoader(trainset_valid, 
                                          batch_size=72, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset_valid.collate_fn)


#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#net = torch.nn.DataParallel(net, device_ids=[0])

net = RetinaNet()
net = torch.nn.DataParallel(net, device_ids=[0])
net.cuda()

id_net = Idnet(classnum=2874)
id_net = torch.nn.DataParallel(id_net, device_ids=[0])
id_net.cuda()

#MCP = arcface_loss2.Arcface(1024, 2874).cuda()
MCP = MarginCosineProduct(1024, 2874).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD([{'params': id_net.parameters()}, {'params':MCP.parameters()}], 
#optimizer = optim.SGD(id_net.parameters(), 
                      lr=1e-3, 
                      momentum=0.9, 
                      weight_decay=1e-4)

net.load_state_dict(torch.load("./trained model/originalFAN_model.pth"))
net.eval()
coder = DataEncoder()

detector = dlib.get_frontal_face_detector()
predicter_path = "./model/shape_predictor_5_face_landmarks.dat"
sp = dlib.shape_predictor(predicter_path)

euclidean_dis = torch.nn.PairwiseDistance()
cosine_dis = torch.nn.CosineSimilarity()


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def train(epoch, file_obj, acc):
    print("\n Epoch : %d" % epoch)
    trainTotalRight = 0
    trainTotalImg = 0
    for batch_idx, (inputs, img_path, ids) in enumerate(trainloader):
        trainTotalImg += inputs.shape[0]
        #id_net.train()
        MCP.train()
        inputs = Variable(inputs.cuda(), volatile=True)
        img_input = torch.zeros(inputs.size(0), 3, 112, 96)
        for img_counter in range(inputs.size(0)):
            sampled = F.upsample(inputs[img_counter, :, :, :].view(1,3,224,224), 
                                 size=(112, 96), 
                                 mode='bilinear')
            img_input[img_counter, :,:,:] = sampled

        inputs = Variable(img_input.cuda())
        optimizer.zero_grad()
        id_net_out = id_net(inputs)
        #pdb.set_trace()
        id_out_shape = id_net_out.size()
        target = torch.tensor(ids).view(id_out_shape[0]).cuda()
        output = MCP(id_net_out, target)
        (_, estimate_id) = torch.max(output, dim=1)
        #batchCorrect = torch.eq(estimate_id, torch.tensor(ids).cuda() ).sum().item()
        print('\ntrain test:')
        print('estimted:', estimate_id[:8])
        print('ids', ids[:8])
        print('\n')
        batchCorrect = torch.eq(estimate_id, target ).sum().item()
        trainTotalRight += batchCorrect
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        file_obj.write('epoch|'+str(epoch)+'|loss|'+str(loss)+''+'\n')
        print('epoch: %d | train_loss: %.3f |' % (epoch, loss))
        #pdb.set_trace()

        ################################################
        ################################################
        ################################################

        if batch_idx%20 == 0:
            trainAcc = 1.0 *trainTotalRight / trainTotalImg
            print("epoch {}|minibatch {}| trainAcc= {}/{} ={:.5f}".format(epoch, batch_idx,  trainTotalRight, trainTotalImg, trainAcc))
            trainTotalRight = 0
            trainTotalImg = 0

            id_net.eval()
            MCP.eval()
            estimate = 0
            for batch_idx, (inputs, img_path, ids) in enumerate(trainloader_valid):
                inputs = Variable(inputs.cuda(), volatile=True)
                img_input = torch.zeros(inputs.size(0), 3, 112, 96)
                for img_counter in range(inputs.size(0)):
                    sampled = F.upsample(inputs[img_counter, :, :, :].view(1,3,224,224), 
                                        size=(112, 96), 
                                        mode='bilinear')
                    img_input[img_counter, :,:,:] = sampled

                inputs = Variable(img_input.cuda())
                with torch.no_grad():
                    id_net_out = id_net(inputs)
                    id_out_shape = id_net_out.size()
                    target = torch.tensor(ids).view(id_out_shape[0]).cuda()
                    output = MCP(id_net_out, target)
                _, estimate_id = torch.max(output, dim=1)
                print('\nevalute test:')
                print('estimate:', estimate_id[:8])
                print('ids:', ids[:8])
                print('\n')
                estimate += sum(torch.eq(estimate_id, torch.tensor(ids).cuda()))
            print("estimate-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#-*#={}".format(estimate.item() ))
            acc_tmp = float(estimate)/100
            print('----------acc:',acc_tmp)
            if acc_tmp > acc and acc>0:
                acc = acc_tmp
                save_model(id_net, 'arcface_id_net-cosface-acc'+str(acc)+'.pth')

        if epoch%20 ==0 and epoch>0:
            save_model(id_net, 'arcface_id_net-cosface-epoch-'+str(epoch)+'-acc'+str(acc)+'.pth')
    return acc


def test():
    pass


if __name__ == '__main__':
    print('----------------arcface 3000-----------------------')
    acc = 0    
    if args.state == 'train':
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if not os.path.exists('./save_file'):
            os.mkdir('./save_file')
        save_file = './save_file/save_file-'+date_now+'.txt'

        file_obj = open(save_file, 'a')
        file_obj.write('description: arcface, softmax3000, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 100, record loss')
        for epoch in range(start_epoch, start_epoch+200):
            acc = train(epoch, file_obj, acc)  
            #test(epoch)
        file_obj.close() 
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')

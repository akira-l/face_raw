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
from datagen import ListDataset, face_a_ListDataset

from torch.autograd import Variable
from encoder import DataEncoder


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='softmax for training')
parser.add_argument('--state', '-s', help='training or test stage')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
global tensorboard_counter
tensorboard_counter = 0

batch_size=8


# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = face_a_ListDataset(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1, 
                                          collate_fn=trainset.collate_fn)

testset = face_a_ListDataset(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         num_workers=1, 
                                         collate_fn=testset.collate_fn)

#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#net = torch.nn.DataParallel(net, device_ids=[0])

net = RetinaNet()
net = torch.nn.DataParallel(net, device_ids=[0])
net.cuda()

id_net = Idnet()
id_net = torch.nn.DataParallel(id_net, device_ids=[0])
id_net.cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(id_net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

net.load_state_dict(torch.load("./trained model/originalFAN_model.pth"))
net.eval()
coder = DataEncoder()


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def train(epoch, file_obj, val_obj):
    print("\n Epoch : %d" % epoch)
    id_net.train()
    for batch_idx, (inputs, img_path, ids) in enumerate(trainloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_preds, cls_preds = net(inputs)
        boxes = []
        for box_counter in range(inputs.size(0)):
            #try:
            box, label, score = coder.decode(loc_preds[box_counter].data.cpu(), cls_preds[box_counter].data.cpu(), (224, 224))
            #except IndexError:
            #box, label, score = coder.decode(loc_preds[box_counter].data.cpu(), cls_preds[box_counter].data.cpu(), (224, 224), debug_flag=True)
            #pdb.set_trace()
            #else:
            #print('check point 1 over')
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
                #print('\n', tmp_box ,'\n')
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
            #print('face_box', face_box)
            #try:
            sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), size=(150, 150), mode='bilinear')
            #except RuntimeError:
            #pdb.set_trace()
            #else:
            #print('check point 2 over')
            img_input[img_counter, :,:,:] = sampled

        inputs = Variable(img_input.cuda())
        optimizer.zero_grad()
        id_net_out = id_net(inputs)

        id_out_shape = id_net_out.size()
        #pdb.set_trace()
        loss = criterion(id_net_out.view(id_out_shape[0], id_out_shape[1]), torch.tensor(ids).view(id_out_shape[0]).cuda())
        loss.backward()
        optimizer.step()

        file_obj.write('epoch|'+str(epoch)+'|loss|'+str(loss)+''+'\n')
        print('epoch: %d | train_loss: %.3f |' % (epoch, loss))
        #pdb.set_trace()
        
        #grid = torchvision.utils.make_grid(img_input, 1)
        #torchvision.utils.save_image(grid, 'a.jpg')
    save_model(id_net, 'softmax_id_net.pth')

def test():
    pass

if __name__ == '__main__':
    if args.state == 'train':
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if not os.path.exists('./save_file'):
            os.mkdir
        save_file = './save_file/save_file-'+date_now+'.txt'

        val_file = './var_log.txt'
        val_obj = open(val_file, 'w')
        file_obj = open(save_file, 'a')
        file_obj.write('description: softmax, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 100, record loss')
        for epoch in range(start_epoch, start_epoch+100):
            train(epoch, file_obj, val_obj)  
            #test(epoch)
        file_obj.close() 
        val_obj.close()   
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')

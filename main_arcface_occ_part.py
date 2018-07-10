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
from datagen import ListDataset, face_a_ListDataset_with_att

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

batch_size=16

# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

transform_att = transforms.Compose([
    transforms.ToTensor()
])

trainset = face_a_ListDataset_with_att(root="./../face_a/train",
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

testset = face_a_ListDataset_with_att(root="./../face_a/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      transform_att=transform_att,
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

MCP = arcface_loss2.Arcface(1024, 4096).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD([{'params': id_net.parameters()}, {'params':MCP.parameters()}], 
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

def train(epoch, file_obj):
    print("\n Epoch : %d" % epoch)
    id_net.train()
    for batch_idx, (inputs, img_path, ids, att) in enumerate(trainloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_preds, cls_preds = net(inputs)
        boxes = []
        for box_counter in range(inputs.size(0)):
            #try:
            box, label, score = coder.decode(loc_preds[box_counter].data.cpu(), 
                                             cls_preds[box_counter].data.cpu(), 
                                             (224, 224))
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
        seg_score = torch.zeros(inputs.size(0), 3)
        img_part1 = torch.zeros(inputs.size(0), 3, 150, 150)
        img_part2 = torch.zeros(inputs.size(0), 3, 150, 150)
        img_part3 = torch.zeros(inputs.size(0), 3, 150, 150)

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
                                 size=(150, 150), 
                                 mode='bilinear')
            att_sampled = F.upsample(att[img_counter, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,1,width,height), 
                                 size=(150, 150), 
                                 mode='bilinear')
            seg_score[img_counter, 0] = sum(sum(att_sampled[0, :, 0:50].view(50,150)))/(150*50)
            seg_score[img_counter, 1] = sum(sum(att_sampled[0, :, 50:100].view(50,150)))/(150*50)
            seg_score[img_counter, 2] = sum(sum(att_sampled[0, :, 100:150].view(50,150)))/(150*50)
            
            img_input[img_counter, :,:,:] = sampled*att_sampled.cuda()
            img_part1[img_counter, :, :, 0:50] = img_input[img_counter, :, :, 0:50]
            img_part2[img_counter, :, :, 50:100] = img_input[img_counter, :, :, 50:100]
            img_part3[img_counter, :, :, 100:150] = img_input[img_counter, :, :, 100:150]

        inputs = Variable(img_input.cuda())

        optimizer.zero_grad()
        id_net_out = id_net(inputs)
        part1_out = id_net(img_part1)
        part2_out = id_net(img_part2)
        part3_out = id_net(img_part3)

        id_out_shape = id_net_out.size()
        #pdb.set_trace()
        target = torch.tensor(ids).view(id_out_shape[0]).cuda()
        output = MCP(id_net_out, target)
        out_part1 = MCP(part1_out, target)
        out_part2 = MCP(part2_out, target)
        out_part3 = MCP(part3_out, target)
        
        loss_part = torch.zeros(inputs.size(0), 3)
        loss = criterion(output, target)
        loss_part[:,0] = criterion(out_part1, target)
        loss_part[:,1] = criterion(out_part2, target)
        loss_part[:,2] = criterion(out_part3, target)
        
        loss_all = loss+torch.sum(loss_part*seg_score).cuda()

        loss_all.backward()
        optimizer.step()

        file_obj.write('epoch|'+str(epoch)+'|loss|'+str(loss_all)+''+'\n')
        print('epoch: %d | train_loss: %.3f |' % (epoch, loss_all))
        #pdb.set_trace()
        
        #grid = torchvision.utils.make_grid(img_input, 1)
        #torchvision.utils.save_image(grid, 'a.jpg')
    save_model(id_net, 'arcface_id_net_occ_part.pth')

def test():
    pass

if __name__ == '__main__':
    if args.state == 'train':
        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        if not os.path.exists('./save_file'):
            os.mkdir
        save_file = './save_file/save_file-'+date_now+'.txt'

        file_obj = open(save_file, 'a')
        file_obj.write('description: arcface, with occlusion parted, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 100, record loss')
        for epoch in range(start_epoch, start_epoch+150):
            train(epoch, file_obj)  
            #test(epoch)
        file_obj.close() 
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')

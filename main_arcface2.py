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
from Idnet3 import Idnet
from datagen2 import *
from eval_datagen import face_a_dis_valid

from torch.autograd import Variable
from encoder import DataEncoder
from arcface_loss2 import Arcface
import cosface_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='softmax for training')
parser.add_argument('--state', '-s', help='training or test stage')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
global tensorboard_counter
tensorboard_counter = 0

batch_size=64


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
                      input_size=224)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset.collate_fn)

trainset_valid = face_a_dis_valid(root="./../face_a/face_image/train",
                      list_file = "./../face_a/train.csv",
                      train=False, 
                      transform=transform, 
                      input_size=224)
trainloader_valid = torch.utils.data.DataLoader(trainset_valid, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=0, 
                                          collate_fn=trainset_valid.collate_fn)


#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#net = torch.nn.DataParallel(net, device_ids=[0])

'''
net = RetinaNet()
net = torch.nn.DataParallel(net, device_ids=[0])
net.cuda()
'''

id_net = Idnet()
id_net = torch.nn.DataParallel(id_net, device_ids=[0])
id_net.cuda()

#MCP = Arcface(512, 2874).cuda()
MCP = cosface_loss.MarginCosineProduct(512, 2874).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()

'''
net.load_state_dict(torch.load("./trained model/originalFAN_model.pth"))
net.eval()
'''
coder = DataEncoder()

#euclidean_dis = torch.nn.PairwiseDistance()
#cosine_dis = torch.nn.CosineSimilarity()
cosine_dis = torch.nn.CosineSimilarity()

def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def train(epoch, file_obj, acc, optimizer):
    print("\n Epoch : %d" % epoch)
    id_net.train()
    for batch_idx, (inputs, img_path, ids) in enumerate(trainloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        '''
        with torch.no_grad():
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
            '''
        
        img_input = torch.zeros(inputs.size(0), 3, 112, 96)
        
        for img_counter in range(inputs.size(0)):
            '''
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
            sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), 
                                 size=(112, 96), 
                                 mode='bilinear')
            #except RuntimeError:
            #pdb.set_trace()
            #else:
            #print('check point 2 over')
            '''
            sampled = F.upsample(inputs[img_counter, :, :, :].view(1,3,224,224), 
                                 size=(112, 96), 
                                 mode='bilinear')
            img_input[img_counter, :,:,:] = sampled

        inputs = Variable(img_input.cuda())
        #print('------------------------------------------------------------------------\ninputs', inputs)
        optimizer.zero_grad()
        id_net_out = id_net(inputs)
        print('------------------------------------------------------------------------\nid_net_out', id_net_out)

        id_out_shape = id_net_out.size()
        target = torch.tensor(ids).view(id_out_shape[0]).cuda()
        #print('------------------------------------------------------------------------\ntarget', target)
        #output = MCP(id_net_out, target)
        #print('------------------------------------------------------------------------\nMCP output', output)
        #pdb.set_trace()
        loss = criterion(id_net_out, target)
        #print('------------------------------------------------------------------------\nloss', loss)
        #pdb.set_trace()
        loss.backward()
        optimizer.step()

        file_obj.write('epoch|'+str(epoch)+'|loss|'+str(loss)+''+'\n')
        print('arcface epoch: %d | train_loss: %.3f |' % (epoch, loss))
        #pdb.set_trace()
        
        ################################################
        ################################################
        ################################################

        if batch_idx%20 == 0:
            id_net.eval()
            estimate = 0
            val_dis = torch.tensor([]).cuda()
            val_flag = []
            for batch_idx, valid_pair in enumerate(trainloader_valid):
                pair_feature = []
                for pair_counter in range(2):
                    inputs = valid_pair[pair_counter]
                    inputs = Variable(inputs.cuda(), volatile=True)
                    '''
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
                            #print('\n', tmp_box ,'\n')
                        boxes.append(tmp_box)
                    '''
                    img_input = torch.zeros(inputs.size(0), 3, 112, 96)
                    for img_counter in range(inputs.size(0)):
                        '''
                        face_box = boxes[img_counter]
                        face_box = [int(x) for x in face_box]
                        face_box[0] = max(face_box[0], 0)
                        face_box[1] = max(face_box[1], 0)
                        face_box[2] = min(face_box[2], inputs.size(2))
                        face_box[3] = min(face_box[3], inputs.size(2))

                        height = face_box[3]-face_box[1]
                        width = face_box[2]-face_box[0]
                        sampled = F.upsample(inputs[img_counter, :, face_box[0]:face_box[2], face_box[1]:face_box[3]].view(1,3,width,height), 
                                            size=(112, 96), 
                                            mode='bilinear')
                        '''
                        sampled = F.upsample(inputs[img_counter, :, :, :].view(1,3,224,224), 
                                 size=(112, 96), 
                                 mode='bilinear')
                        img_input[img_counter, :,:,:] = sampled

                    inputs = Variable(img_input.cuda())
                    with torch.no_grad():
                        id_net_out = id_net(inputs)

                    pair_feature.append(id_net_out)
                
                dis = cosine_dis(pair_feature[0], pair_feature[1])
                val_dis = torch.cat((val_dis, dis), 0)
                val_flag = val_flag+valid_pair[2]
            print('\ndistance cmp:', val_dis)
            val_flag = torch.tensor(val_flag).cuda()
            pos_thresh = min(val_dis*val_flag)
            neg_thresh = max(val_dis*abs(val_flag-1))
            thresh = 0
            if pos_thresh >= neg_thresh:
                check_thresh = neg_thresh
                check_step = (pos_thresh-neg_thresh)/20
                acc_tmp = 0
                while check_thresh < pos_thresh:
                    check_estimate = ( sum((val_dis<=check_thresh).float()*val_flag)+ \
                                       sum((val_dis>check_thresh).float()*abs(check_thresh)) )/100
                    check_estimate = float(check_estimate)
                    if check_estimate > acc_tmp:
                        acc_tmp = check_estimate
                        thresh = check_thresh
                    check_thresh += check_step
            else:
                acc_tmp = 1.0
                thresh = (neg_thresh+pos_thresh)/2
            print('check thresh: %f' % thresh)
            print('\n----------acc:',acc_tmp)
            #pdb.set_trace()
            if acc_tmp > acc and acc>0:
                acc = acc_tmp
                save_model(id_net, 'arcface_id_net-softmax30000-sphere-acc'+str(acc)+'.pth')
    return acc

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
        file_obj.write('description: cosface, optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4), epoch: 100, record loss')
        lr = 0.001
        epoch_total = 100
        for epoch in range(epoch_total):
            optimizer = optim.SGD([{'params': id_net.parameters()}, {'params':MCP.parameters()}], 
                      lr=lr*(0.5**(epoch//50)), 
                      momentum=0.9, 
                      weight_decay=1e-4)

            acc = train(epoch, file_obj, acc, optimizer)  
            #test(epoch)
        file_obj.close()  
    elif args.state == 'test':
        test()
    else:
        raise ValueError('need training or test state input')

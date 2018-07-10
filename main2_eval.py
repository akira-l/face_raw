from __future__ import print_function

import csv, os, random, pdb
import sys
import argparse
import datetime
import cv2
from PIL import Image
import numpy as np
import dlib

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from torchvision.transforms import functional as F
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

cudnn.benchmark = True

id_net = Idnet(classnum=2874)
id_net = torch.nn.DataParallel(id_net, device_ids=[0])
id_net.load_state_dict(torch.load("./arcface_id_net-data_addition-epoch-20-acc0.pth"))
id_net.cuda()

#net.load_state_dict(torch.load("./trained model/originalFAN_model.pth"))
#net.eval()
coder = DataEncoder()

detector = dlib.get_frontal_face_detector()
predicter_path = "./model/shape_predictor_5_face_landmarks.dat"
sp = dlib.shape_predictor(predicter_path)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds:(i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(float(d[3])))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def alignment(img):
    img_cv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    dets = detector(img_cv, 1)
    if len(dets)!=0:
        faces = dlib.full_object_detections()
        for det in dets:
            faces.append(sp(img_cv, det))
        face_img = dlib.get_face_chips(img_cv, faces, size=160)
        face_img = cv2.resize(face_img[0], (112, 96), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB))
    else:
        img_cv = cv2.resize(img_cv, (112, 96), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))


def get_feature():
    predicts = []
    id_net.eval()
    gallery_root = './../face_a/face_image/gallery'
    probe_root = './../face_a/face_image/probe'
    gallery_list = './../face_a/test_a_gallery.csv'
    probe_list = './../face_a/test_a_probe.csv'

    gallery_file_list = list(csv.reader(open(gallery_list,'r')))
    probe_file_list = list(csv.reader(open(probe_list, 'r')))

    gallery_name = []
    gallery_ids = []
    for gallery in gallery_file_list:
        gallery_name.append(gallery[0])
        gallery_ids.append(gallery[1])
        
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    gallery_feature = torch.zeros(len(gallery_ids), 1024)
    for i in range(len(gallery_ids)):
        name = gallery_name[i]
        img = Image.open(os.path.join(gallery_root, name)).convert('RGB')
        img = alignment(img)
        img, img_ = transform(img), transform(F.hflip(img))
        img, img_ = Variable(img.unsqueeze(0).cuda(), volatile=True), Variable(img_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        #pdb.set_trace()
        print(i)
        face_feature = torch.cat((id_net(img), id_net(img_)), 1).data.cpu()[0]
        gallery_feature[i,:] = face_feature  
    return gallery_feature, gallery_ids

def gen_csv(gallery_feature, gallery_ids):
    predicts = []
    id_net.eval()
    probe_root = './../face_a/face_image/probe'
    probe_list = './../face_a/test_a_probe.csv'

    probe_name = list(csv.reader(open(probe_list, 'r')))
        
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    csv_file = open('arcface_addition10.csv', 'w+')
    csvwriter = csv.writer(csv_file, lineterminator='\n')
    id_ = []
    for i in range(len(probe_name)):
        name = probe_name[i][0]
        img = Image.open(os.path.join(probe_root, name)).convert('RGB')
        img = alignment(img)
        img, img_ = transform(img), transform(F.hflip(img))
        img, img_ = Variable(img.unsqueeze(0).cuda(), volatile=True), Variable(img_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        face_feature = torch.cat((id_net(img), id_net(img_)), 1).data.cpu()[0]
        dis = []
        for gallery_counter in range(gallery_feature.size(0)):
            f1 = gallery_feature[gallery_counter, :]
            f2 = face_feature
            cos_dis = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            dis.append(float(cos_dis))
        id_num = dis.index(max(dis))
        id_.append(str(gallery_ids[id_num]))
    
    
    for csv_counter in range(len(id_)):
        row_content = [[probe_name[csv_counter][0], id_[csv_counter]]]
        csvwriter.writerows(row_content)
    
def test_eval():
    id_net.eval()
    fnames = []
    ids = []
    ids_list = list(range(2874))
    im_name_list = []
    root = "./../face_a/train"
    encoder = DataEncoder()
    list_file = "./../face_a/train.csv"
    file_list = csv.reader(open(list_file,'r'))
    file_list = list(file_list)
    # 2874
    for content_counter in range(len(file_list)):
        fnames.append(os.path.join(root, file_list[content_counter][0]))
        ids.append(int(file_list[content_counter][1]))
    
    for id_counter in range(2874):
        seq_num = ids.index(id_counter)
        im_name_list.append(fnames[seq_num])
        del(ids[seq_num])
        del(fnames[seq_num])

    im_name_valid = fnames[:400]
    im_name_train = fnames[400:]+im_name_list
    ids_valid = ids[:400]
    ids_train = ids[400:]+ids_list

    eval_list_feature = torch.zeros(len(ids_list), 1024)
    for i in range(len(ids_list)):
        name = im_name_list[i]
        img = Image.open(name).convert('RGB')
        img = alignment(img)
        img, img_ = transform(img), transform(F.hflip(img))
        img, img_ = Variable(img.unsqueeze(0).cuda(), volatile=True), Variable(img_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        print(i)
        face_feature = torch.cat((id_net(img), id_net(img_)), 1).data.cpu()[0]
        eval_list_feature[i,:] = face_feature  
    
    id_ = []
    for i in range(len(ids_valid)):
        #pdb.set_trace()
        name = im_name_valid[i]
        img = Image.open(name).convert('RGB') 
        
        img = alignment(img)
        img, img_ = transform(img), transform(F.hflip(img))
        img, img_ = Variable(img.unsqueeze(0).cuda(), volatile=True), Variable(img_.unsqueeze(0).cuda(),
                                                                                  volatile=True)
        face_feature = torch.cat((id_net(img), id_net(img_)), 1).data.cpu()[0]
        dis = []
        for gallery_counter in range(eval_list_feature.size(0)):
            f1 = eval_list_feature[gallery_counter, :]
            f2 = face_feature
            cos_dis = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            dis.append(float(cos_dis))
        id_num = dis.index(max(dis))
        id_.append(str(ids_list[id_num]))
    pdb.set_trace()
    acc_counter =0
    for id_counter in range(len(id_)):
        if id_[id_counter] == ids_valid[id_counter]:
            acc_counter +=1
    print(acc_counter/400.0)
    


    



if __name__ == '__main__':
    #test_eval()
    
    feature, id_list = get_feature()
    pdb.set_trace()
    gen_csv(feature, id_list)
    

from matplotlib.pyplot import xticks
from numpy.lib.function_base import angle
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torchvision import datasets, models, transforms
import torch.optim as optim
from early_stopping_pytorch.pytorchtools import EarlyStopping

from models.resnet import resnet50
import models.resnet_fpn as resnet_fpn

import wandb

wandb.init(project="Retail", entity="shashimalcse")

class GazeOptimizer():
    def __init__(self, net, initial_lr, weight_decay=1e-6):
        
        self.INIT_LR = initial_lr
        self.WEIGHT_DECAY = weight_decay
        self.optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    def getOptimizer(self, epoch, decay_epoch=15):
        
        if epoch < decay_epoch:
            lr = self.INIT_LR
        else:
            lr = self.INIT_LR / 10

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = self.WEIGHT_DECAY
            
        return self.optimizer




class Shashimal6_Face3D(nn.Module):
    def __init__(self):
        super(Shashimal6_Face3D,self).__init__()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet50(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 3)   
    def forward(self,image,face):
        self.depth.eval()
        with torch.no_grad():
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        return output,id

class Shashimal6_Face3D_Student(nn.Module):
    def __init__(self):
        super(Shashimal6_Face3D_Student,self).__init__()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Small")
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 3)   
    def forward(self,image,face):
        self.depth.eval()
        with torch.no_grad():
            id = self.depth(image)
            id = torch.nn.functional.interpolate(id.unsqueeze(1),size=image.shape[2:],mode="bicubic",align_corners=False,)
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        return output,id

def get_bb_binary(box):
    xmin, ymin, xmax, ymax = box
    b = np.zeros((224, 224), dtype='float32')
    for j in range(ymin, ymax):
        for k in range(xmin, xmax):
            b[j][k] = 1
    return b


def train_face3d_distill(student_model,teacher_model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
    alpha=0.3
    for epoch in range(num_epochs):

        student_model.train()
        teacher_model.eval() 
        running_loss = []
        running_loss2 = []
        student_losses = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            with torch.no_grad():
                gaze_t,depth_t = teacher_model(image,face)
            gaze,depth = student_model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            student_loss = criterion(gaze, label)
            ditillation_loss = divergence_loss_fn(F.softmax(gaze, dim=1),F.softmax(gaze_t, dim=1))
            loss = alpha * student_loss + (1 - alpha) * ditillation_loss
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            running_loss2.append(loss.item())
            student_losses.append(student_loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                running_loss = []
        wandb.log({"train_loss": np.mean(running_loss2)})
        wandb.log({"student_train_loss": np.mean(student_losses)})
        student_model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image = img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze, depth = student_model(image, face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0], 3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i]) / 224
                label[i, 0] = gt[0]
                label[i, 1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)
        logger.info('%s'%(str(val_loss)))
        wandb.log({"student_train_loss": val_loss})
        validation_loss = []
        early_stopping(val_loss, student_model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return student_model


def train_face3d(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        running_loss2 = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze,depth = model(image,face)
            depth =  depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy()*224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy()*224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary==1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary==1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            running_loss2.append(loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                running_loss = []
        wandb.log({"train_loss": np.mean(running_loss2)})
        model.eval()
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap, head_box, gtbox) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image = img.cuda()
            face = face.cuda()
            gt_label = gt_label
            head = head
            optimizer.zero_grad()
            gaze, depth = model(image, face)
            depth = depth.cpu()
            max_depth = torch.max(depth)
            depth = depth / max_depth
            head_box = head_box.cpu().detach().numpy() * 224
            head_box = head_box.astype(int)
            gtbox = gtbox.cpu().detach().numpy() * 224
            gtbox = gtbox.astype(int)
            label = np.zeros((image.shape[0], 3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i]) / 224
                label[i, 0] = gt[0]
                label[i, 1] = gt[1]
                hbox_binary = torch.from_numpy(get_bb_binary(head_box[i]))
                gtbox_binary = torch.from_numpy(get_bb_binary(gtbox[i]))
                hbox_depth = torch.mul(depth[i], hbox_binary)
                gtbox_depth = torch.mul(depth[i], gtbox_binary)
                head_depth = torch.sum(hbox_depth) / torch.sum(hbox_binary == 1)
                gt_depth = torch.sum(gtbox_depth) / torch.sum(gtbox_binary == 1)
                label[i, 2] = (gt_depth - head_depth)
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)
        logger.info('%s'%(str(val_loss)))
        wandb.log({"val_loss": val_loss})
        validation_loss = []
        early_stopping(val_loss, model, optimizer, epoch, logger)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model



def test_face3d(model, test_data_loader, logger, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap in test_data_loader:
            image =  img.cuda()
            face = face.cuda()
            gaze,depth = model(image,face)
            depth =  depth.cpu().data.numpy()
            gaze =  gaze.cpu().data.numpy()
            label = np.zeros((image.shape[0],3))
            for i in range(image.shape[0]):
                gt = (gt_label[i] - head[i])/224
                label[i,0] = gt[0]
                label[i,1] = gt[1]
                label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])/224
            for i in range(img.shape[0]):
                ae = np.arccos(np.dot(gaze[i,:2],label[i,:2])/np.sqrt(np.dot(label[i,:2],label[i,:2])*np.dot(gaze[i,:2],gaze[i,:2])))
                ae = np.maximum(np.minimum(ae,1.0),-1.0) * 180 / np.pi
                angle_error.append(ae)
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)   



 







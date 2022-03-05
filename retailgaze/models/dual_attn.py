from matplotlib.pyplot import xticks
from numpy.lib.function_base import angle
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import math
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from resnest.torch import resnest50
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torchvision import datasets, models, transforms
import torch.optim as optim
from early_stopping_pytorch.pytorchtools import EarlyStopping

from models.resnet import resnet18
from models.resnet import resnet50
import models.resnet_fpn as resnet_fpn

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # bottom up
        self.resnet = resnet_fpn.resnet50(pretrained=True)

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c5_conv = nn.Conv2d(2048, 256, (1, 1))
        self.c4_conv = nn.Conv2d(1024, 256, (1, 1))
        self.c3_conv = nn.Conv2d(512, 256, (1, 1))
        self.c2_conv = nn.Conv2d(256, 256, (1, 1))
        #self.max_pool = nn.MaxPool2d((1, 1), stride=2)

        self.p5_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p4_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p3_conv = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.p2_conv = nn.Conv2d(256, 256, (3, 3), padding=1)

        # predict heatmap
        self.sigmoid = nn.Sigmoid()
        self.predict = nn.Conv2d(256, 1, (3, 3), padding=1)
 
    def top_down(self, x):
        c2, c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p4 = self.upsample(p5) + self.c4_conv(c4)
        p3 = self.upsample(p4) + self.c3_conv(c3)
        p2 = self.upsample(p3) + self.c2_conv(c2)

        p5 = self.relu(self.p5_conv(p5))
        p4 = self.relu(self.p4_conv(p4))
        p3 = self.relu(self.p3_conv(p3))
        p2 = self.relu(self.p2_conv(p2))

        return p2, p3, p4, p5

    def forward(self, x):
        # bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # top down
        p2, p3, p4, p5 = self.top_down((c2, c3, c4, c5))

        heatmap = self.sigmoid(self.predict(p2))
        return heatmap

class GazeStatic(nn.Module):
    def __init__(self):
        super(GazeStatic, self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 3)


    def forward(self, x_in):

        base_out = self.base_model(x_in)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)


        angular_output = output[:,:2]
        angular_output[:,0:1] = math.pi*nn.Tanh()(angular_output[:,0:1])
        angular_output[:,1:2] = (math.pi/2)*nn.Tanh()(angular_output[:,1:2])

        var = math.pi*nn.Sigmoid()(output[:,2:3])
        var = var.view(-1,1).expand(var.size(0), 2)

        return angular_output,var

class Shashimal6(nn.Module):
    def __init__(self):
        super(Shashimal6,self).__init__()
        self.compress_conv1 = nn.Conv2d(2051, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        resnet =  resnet50(pretrained=True)
        self.scene_net = torch.nn.Sequential(*(list(resnet.children())[:-3]))
        self.face_net = Shashimal6_Face3D()
        statedict = torch.load("/content/drive/MyDrive/shashimal6_face_51.pt")
        self.face_net.cuda()
        self.face_net.load_state_dict(statedict["state_dict"])
        self.sigmoid = nn.Sigmoid()    
        self.linear = nn.Linear(1,1)
        self.linear.weight.data.fill_(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self,image,face,object_channel,head_channel,head_point):
        self.face_net.eval()
        with torch.no_grad():
            gaze,depth= self.face_net(image,face)
        #     fd_range = torch.zeros(image.shape[0],1).cuda()
        #     head_depth = torch.zeros(image.shape[0],1).cuda()
        #     for batch in range(image.shape[0]):
        #         fd_range[batch,:] = (torch.max(depth[batch]) - torch.min(depth[batch]))/24
        #         head_depth[batch,:] = depth[batch,:,head_point[batch,0],head_point[batch,1]]
        #     point_depth = torch.zeros(image.shape[0],1).cuda()
        # for batch in range(image.shape[0]):
        #     point_depth[batch,:] = head_depth[batch] + gaze[batch,2]*224   
        # fd_0 = torch.zeros(image.shape[0],1,224,224).cuda()
        # fd_1 = torch.zeros(image.shape[0],1,224,224).cuda()
        # fd_2 = torch.zeros(image.shape[0],1,224,224).cuda()
        # for batch in range(image.shape[0]):
        #     fd_0[batch,:,:,:] = torch.where((point_depth[batch]-fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
        #     fd_1[batch,:,:,:] = torch.where((point_depth[batch]-2*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+2*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
        #     fd_2[batch,:,:,:] = torch.where((point_depth[batch]-3*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+3*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
        xy = gaze[:,:2]
        xy = xy.float()
        head_point = head_point.float()
        mask = torch.zeros(image.shape[0],1,224,224).cuda()
        for batch in range(image.shape[0]):
            for i in range(224):
                for k in range(224):
                    arr = torch.tensor([k,i],dtype=torch.float32).cuda() - head_point[batch,:]
                    arr = arr.float()
                    mask[batch,:,i,k] = torch.dot(arr,xy[batch,:])/(torch.norm(arr,p=2)*torch.norm(xy[batch,:],p=2))
        mask = torch.arccos(mask)
        mask = torch.maximum(1-(12*mask/np.pi),torch.tensor(0))
        mask = torch.nan_to_num(mask)
        # x_0 = torch.mul(fd_0,mask)
        # x_1 = torch.mul(fd_1,mask)
        # x_2 = torch.mul(fd_2,mask)
        # depth_mask_0 = torch.mul(x_0,object_channel)
        # depth_mask_1 = torch.mul(x_1,object_channel)
        # depth_mask_2 = torch.mul(x_2,object_channel)
        # depth_mask = torch.cat([depth_mask_0,depth_mask_1,depth_mask_2], dim=1)   
        # scene = self.scene_net(image)
        # reduce_depth_mask = self.maxpool(self.maxpool(self.maxpool(self.maxpool(self.maxpool(depth_mask)))))
        # scene_depth_feat = torch.cat([scene,reduce_depth_mask],1)
        # encoding = self.compress_conv1(scene_depth_feat)
        # encoding = self.compress_bn1(encoding)
        # encoding = self.relu(encoding)
        # encoding = self.compress_conv2(encoding)
        # encoding = self.compress_bn2(encoding)
        # encoding = self.relu(encoding)

        # x = self.deconv1(encoding)
        # x = self.deconv_bn1(x)
        # x = self.relu(x)
        # x = self.deconv2(x)
        # x = self.deconv_bn2(x)
        # x = self.relu(x)
        # x = self.deconv3(x)
        # x = self.deconv_bn3(x)
        # x = self.relu(x)
        # x = self.conv4(x)
        
        return mask

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def d(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)
def train(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel,head,gt_label,gaze_heatmap) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_point = head.cuda()
            gaze_heatmap = gaze_heatmap.cuda().to(torch.float)
            optimizer.zero_grad()
            heatmap = model(image,face,object_channel,head_point)
            heatmap = heatmap.squeeze(1)
            loss = criterion(heatmap,gaze_heatmap)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = [] 
        model.eval() 
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_point = head.cuda()
            gaze_heatmap = gaze_heatmap.cuda().to(torch.float)
            optimizer.zero_grad()
            heatmap = model(image,face,object_channel,head_point)
            heatmap = heatmap.squeeze(1)
            loss = criterion(heatmap,gaze_heatmap)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)
        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []
        early_stopping(val_loss, model, optimizer, epoch, logger)  
        if early_stopping.early_stop:
            print("Early stopping")
            break                 




        
        
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



class Shashimal6_Face(nn.Module):
    def __init__(self):
        super(Shashimal6_Face,self).__init__()
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet50(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, 2)   
    def forward(self,face):
        base_out = self.base_model(face)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)
        return output
        
def train_face(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            face = face.cuda()
            gt_label = gt_label.cuda()
            head = head.cuda()
            optimizer.zero_grad()
            gaze = model(face)
            gt = (gt_label - head)/224
            loss = criterion(gt,gaze)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                # writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = [] 
        model.eval() 
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            face = face.cuda()
            gt_label = gt_label.cuda()
            head = head.cuda()
            optimizer.zero_grad()
            gaze = model(face)
            gt = (gt_label - head)/224
            loss = criterion(gt,gaze)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)
        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
        validation_loss = []
        early_stopping(val_loss, model, optimizer, epoch, logger)  
        if early_stopping.early_stop:
            print("Early stopping")
            break  

    return model
def test_face(model, test_data_loader, logger, save_output=False):
    model.eval()
    angle_error = []
    with torch.no_grad():
        for img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap in test_data_loader:
            face = face.cuda()
            gt_label = gt_label
            head = head
            gaze = model(face)
            gaze = gaze.cpu().data.numpy()
            gt = (gt_label - head)/224
            for i in range(img.shape[0]):
                ae = np.arccos(np.dot(gaze[i,:],gt[i,:])/np.sqrt(np.dot(gt[i,:],gt[i,:])*np.dot(gaze[i,:],gaze[i,:])))
                ae = np.maximum(np.minimum(ae,1.0),-1.0) * 180 / np.pi
                angle_error.append(ae)
            
        angle_error = np.mean(np.array(angle_error),axis=0)
    print(angle_error)                


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


def get_bb_binary(box):
    xmin, ymin, xmax, ymax = box
    b = np.zeros((224, 224), dtype='float32')
    for j in range(ymin, ymax):
        for k in range(xmin, xmax):
            b[j][k] = 1
    return b


def train_face3d(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
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
                # label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                # writer.add_scalar('training_loss',np.mean(running_loss),epoch*n_total_steps+i)
                running_loss = []


         # Validation
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
                label[i,2] = (depth[i,:,gt_label[i,0],gt_label[i,1]] - depth[i,:,head[i,0],head[i,1]])
            label = torch.tensor(label, dtype=torch.float)
            gaze = gaze.cpu()
            loss = criterion(gaze, label)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)

        logger.info('%s'%(str(val_loss)))
        writer.add_scalar('validation_loss',val_loss,epoch)
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

def save_tensor(model,train_data_loader,validation_data_loader,test_data_loader):
    model.eval()
    for i, (img, face, head_channel,object_channel,fov,eye,head_for_mask,gaze_heatmap, image_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
        image =  img.cuda()
        face = face.cuda()
        object_channel = object_channel.cuda()
        head_point = head_for_mask.cuda()
        heatmap = model(image,face,object_channel,head_channel,head_point)
        heatmap = heatmap.cpu()
        for batch in range(img.shape[0]):
            path = image_path[batch]
            path = path.split('/')
            path[-1] = path[-1].split('.')[0]
            path = "".join(path[-3:])
            torch.save(heatmap[batch],'/content/drive/MyDrive/RetailGaze/masks/{}'.format(path))
    for i, (img, face, head_channel,object_channel,fov,eye,head_for_mask,gaze_heatmap, image_path) in tqdm(enumerate(validation_data_loader), total=len(validation_data_loader)) :
        image =  img.cuda()
        face = face.cuda()
        object_channel = object_channel.cuda()
        head_point = head_for_mask.cuda()
        heatmap = model(image,face,object_channel,head_channel,head_point)  
        heatmap = heatmap.cpu()
        for batch in range(img.shape[0]):
            path = image_path[batch]
            path = path.split('/')
            path[-1] = path[-1].split('.')[0]
            path = "".join(path[-3:])
            torch.save(heatmap[batch],'/content/drive/MyDrive/RetailGaze/masks/{}'.format(path))
    for i, (img, face, head_channel,object_channel,fov,eye,head_for_mask,gaze_heatmap, image_path) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)) :
        image =  img.cuda()
        face = face.cuda()
        object_channel = object_channel.cuda()
        head_point = head_for_mask.cuda()
        heatmap = model(image,face,object_channel,head_channel,head_point)         
        heatmap = heatmap.cpu()
        for batch in range(img.shape[0]):
            path = image_path[batch]
            path = path.split('/')
            path[-1] = path[-1].split('.')[0]
            path = "".join(path[-3:])
            torch.save(heatmap[batch],'/content/drive/MyDrive/RetailGaze/masks/{}'.format(path))


class Shashimal6_New(nn.Module):
    def __init__(self):
        super(Shashimal6_New,self).__init__()
        self.compress_conv1 = nn.Conv2d(2051, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        resnet =  resnet50(pretrained=True)
        self.scene_net = torch.nn.Sequential(*(list(resnet.children())[:-3]))
        self.depth = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.face_net = Shashimal6_Face3D()
        statedict = torch.load("/content/drive/MyDrive/shashimal6_face_51.pt")
        self.face_net.cuda()
        self.face_net.load_state_dict(statedict["state_dict"])
        self.sigmoid = nn.Sigmoid()    
        self.linear = nn.Linear(1,1)
        self.linear.weight.data.fill_(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self,image,face,object_channel,head_point,mask):
        # self.depth.eval()
        self.face_net.eval()
        with torch.no_grad():
            gaze,depth= self.face_net(image,face)
            fd_range = torch.zeros(image.shape[0],1).cuda()
            head_depth = torch.zeros(image.shape[0],1).cuda()
            for batch in range(image.shape[0]):
                fd_range[batch,:] = (torch.max(depth[batch]) - torch.min(depth[batch]))/24
                head_depth[batch,:] = depth[batch,:,head_point[batch,0],head_point[batch,1]]
            point_depth = torch.zeros(image.shape[0],1).cuda()
        gaze[:,2] = self.linear(gaze[:,2])
        for batch in range(image.shape[0]):
            point_depth[batch,:] = head_depth[batch] + gaze[batch,2]*224   
        fd_0 = torch.zeros(image.shape[0],1,224,224).cuda()
        fd_1 = torch.zeros(image.shape[0],1,224,224).cuda()
        fd_2 = torch.zeros(image.shape[0],1,224,224).cuda()
        for batch in range(image.shape[0]):
            fd_0[batch,:,:,:] = torch.where((point_depth[batch]-fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
            fd_1[batch,:,:,:] = torch.where((point_depth[batch]-2*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+2*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
            fd_2[batch,:,:,:] = torch.where((point_depth[batch]-3*fd_range[batch]<=depth[batch,:,:,:]) & (point_depth[batch]+3*fd_range[batch]>=depth[batch,:,:,:]),depth[batch,:,:,:],torch.tensor(0,dtype=torch.float).cuda())
        x_0 = torch.mul(fd_0,mask)
        x_1 = torch.mul(fd_1,mask)
        x_2 = torch.mul(fd_2,mask)
        depth_mask_0 = torch.mul(x_0,object_channel)
        depth_mask_1 = torch.mul(x_1,object_channel)
        depth_mask_2 = torch.mul(x_2,object_channel)
        depth_mask = torch.cat([depth_mask_0,depth_mask_1,depth_mask_2], dim=1)   
        scene = self.scene_net(image)
        reduce_depth_mask = self.maxpool(self.maxpool(self.maxpool(self.maxpool(self.maxpool(depth_mask)))))
        scene_depth_feat = torch.cat([scene,reduce_depth_mask],1)
        encoding = self.compress_conv1(scene_depth_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x

def train_new(model,train_data_loader,validation_data_loader, criterion, optimizer, logger, writer ,num_epochs=5,patience=10):
    since = time.time()
    n_total_steps = len(train_data_loader)
    n_total_steps_val = len(validation_data_loader)
    mse_loss = nn.MSELoss(reduce=False)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):

        model.train()  # Set model to training mode

        running_loss = []
        validation_loss = []
        for i, (img, face, head_channel,object_channel,fov, eye,head_for_mask,gaze_heatmap, image_path) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_point = eye.cuda()
            fov = fov.cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            heatmap = model(image,face,object_channel,head_point,fov)
            heatmap = heatmap.squeeze(1)
            loss = mse_loss(heatmap,gaze_heatmap)
            loss = torch.mean(loss, dim=1)
            loss = torch.mean(loss, dim=1)
            loss = torch.sum(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss.append(loss.item())
            if i % 10 == 9:
                logger.info('%s'%(str(np.mean(running_loss))))
                running_loss = [] 
        model.eval() 
        for i, (img, face, head_channel,object_channel,fov, eye,head_for_mask,gaze_heatmap, image_path) in tqdm(enumerate(validation_data_loader), total=len(train_data_loader)) :
            image =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_point = eye.cuda()
            fov = fov.cuda()
            gaze_heatmap = gaze_heatmap.cuda().to(torch.float)
            heatmap = model(image,face,object_channel,head_point,fov)
            heatmap = heatmap.squeeze(1)
            loss = mse_loss(heatmap,gaze_heatmap)
            loss = torch.mean(loss, dim=1)
            loss = torch.mean(loss, dim=1)
            loss = torch.sum(loss)
            validation_loss.append(loss.item())
        val_loss = np.mean(validation_loss)
        logger.info('%s'%(str(val_loss)))
        validation_loss = []
        early_stopping(val_loss, model, optimizer, epoch, logger)  
        if early_stopping.early_stop:
            print("Early stopping")
            break 


def test_new(model,test_data_loader,logger):
    model.eval()
    total_error = []
    with torch.no_grad():
        for i, (img, face, location_channel,object_channel,head_channel,head,gt_label,gaze_heatmap,mask) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            image =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_point = head.cuda()
            mask = mask.cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            heatmap = model(image,face,object_channel,head_point,mask)
            heatmap = heatmap.squeeze(1) 
            heatmap = heatmap.cpu().data.numpy()
            gaze_heatmap = gaze_heatmap.cpu().data.numpy()
            gt_label = gt_label.cpu().data.numpy()
            head = head.cpu().data.numpy()
            for batch in range(img.shape[0]):
                output = heatmap[batch]
                target = gaze_heatmap[batch]
                gt = gt_label[batch]/[224,224]
                head_position = head[batch]/[224,224]
                h_index, w_index = np.unravel_index(output.argmax(), output.shape)
                f_point = np.array([w_index / 64, h_index / 64])
                f_error = f_point - gt
                f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)
                f_direction = f_point - head_position
                gt_direction = gt - head_position
                ae = np.arccos(np.dot(f_direction,gt_direction)/np.sqrt(np.dot(f_direction,f_direction)*np.dot(gt_direction,gt_direction)))
                ae = np.maximum(np.minimum(ae,1.0),-1.0) * 180 / np.pi
                total_error.append([f_dist, ae])
        l2, ang = np.mean(np.array(total_error), axis=0)
        print(l2,ang)

def test_new_iou(model,test_data_loader,logger):
    model.eval()
    all_iou = []
    with torch.no_grad():
        for i, (img, face, location_channel,object_channel,head_channel ,head,gt_label,gaze_heatmap,mask,gt_bboxes,gt_labels,gaze_idx) in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            img =  img.cuda()
            face = face.cuda()
            object_channel = object_channel.cuda()
            head_channel = head_channel.cuda()
            head_point = head.cuda()
            mask = mask.cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            heatmap = model(img,face,object_channel,head_point,mask)
            heatmap = heatmap.squeeze(1) 
            heatmap = heatmap.cpu().data.numpy()
            gaze_heatmap = gaze_heatmap.cpu().data.numpy()
            gt_label = gt_label.cpu().data.numpy()
            head = head.cpu().data.numpy()
            for batch in range(img.shape[0]):
                output = heatmap[batch]
                max_id = -1
                max_iou = 0
                for k, b in enumerate(gt_bboxes[0]):
                    b = b * [64, 64, 64, 64]
                    b = b.astype(int)
                    iou = np.sum(output[b[1]:b[3],b[0]:b[2]])
                    if iou > max_iou:
                        max_iou = iou
                        max_id = k
                # nearest box by box_iou
                if max_id == -1:
                    all_iou.append(0)
                elif (gaze_idx[batch] == max_id):
                    all_iou.append(1)
                else:
                    all_iou.append(0)

        iou_auc = (sum(all_iou) / len(all_iou)) * 100
        print (iou_auc)
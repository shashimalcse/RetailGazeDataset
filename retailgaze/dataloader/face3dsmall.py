import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from models.gazenet import GazeNet

import time
import os
import numpy as np
import json
import cv2
from PIL import Image, ImageOps
import random
from tqdm import tqdm
import operator
import itertools
from scipy.io import  loadmat
import logging

from scipy import signal
import matplotlib.pyplot as plt
from utils import get_paste_kernel, kernel_map
from pytorchcvtcolor.image import image_to_tensor

import pickle
from skimage import io
from dataloader import chong_imutils

import pandas as pd
np.random.seed(1)
def _get_transform(input_resolution):
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)
def _get_transform2():
    transform_list = []
    transform_list.append(transforms.Resize((448, 448)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)
def boxes2centers(normalized_boxes):
    center_x = (normalized_boxes[:,0] + normalized_boxes[:,2]) / 2
    center_y = (normalized_boxes[:,1] + normalized_boxes[:,3]) / 2
    center_x = np.expand_dims(center_x, axis=1)
    center_y = np.expand_dims(center_y, axis=1)
    normalized_centers = np.hstack((center_x, center_y))
    return normalized_centers  
class GooDataset(Dataset):
    def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=True):
        assert (training in set(['train', 'test']))
        self.root_dir = root_dir
        self.mat_file = mat_file
        self.training = training
        self.include_path = include_path
        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow
        self.transform = _get_transform(input_size)
        self.transform2 = _get_transform2()
        self.use_gtbox= use_gtbox

        with open(mat_file, 'rb') as f:
            self.data = pickle.load(f)
            self.image_num = len(self.data)

        print("Number of Images:", self.image_num)
        logging.info('%s contains %d images' % (self.mat_file, self.image_num))

    def create_mask(self, seg_idx, width=640, height=480):
        seg_idx = seg_idx.astype(np.int64)
        seg_mask = np.zeros((height,width)).astype(np.uint8)
        for i in range(seg_idx.shape[0]):
            seg_mask[seg_idx[i,1],seg_idx[i,0]] = 255
        return seg_mask
    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):

        gaze_inside = True
        data = self.data[idx]
        image_path = data['filename']
        image_path = os.path.join(self.root_dir, image_path)
        #print(image_path)

        eye = [float(data['hx'])/640, float(data['hy'])/480]
        gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
        eyess = np.array([eye[0],eye[1]]).astype(np.float)
        gaze_x, gaze_y = gaze

        image_path = image_path.replace('\\', '/')
        img = Image.open(image_path)
        img = img.convert('RGB')
        width, height = img.size
        #Get bounding boxes and class labels as well as gt index for gazed object
        gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
        gt_labels = np.expand_dims(gt_labels, axis=0)
        gaze_idx = np.copy(data['gazeIdx']).astype(np.int64) #index of gazed object
        gaze_class = np.copy(data['gaze_item']).astype(np.int64) #class of gazed object
        if self.use_gtbox:
            gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
            gt_labels = np.copy(data['ann']['labels'])

            gtbox = gt_bboxes[gaze_idx]
        
        x_min, y_min, x_max, y_max = gt_bboxes[-1] * [width,height,width,height]
        centers = (boxes2centers(gt_bboxes)*[224,224]).astype(int)
        location_channel = np.zeros((224,224), dtype=np.float32)
        for cen in centers:
            location_channel[cen[1],cen[0]] = 1
        head = centers[-1,:]
        gt_label = centers[gaze_idx,:]    
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        
        
        heatmap = get_paste_kernel((224 // 4, 224 // 4), gaze, kernel_map, (224 // 4, 224 // 4))
        seg = data['seg']
        seg_mask = self.create_mask(np.array(seg).astype(np.int64))
        seg_mask = cv2.resize(seg_mask, (224//4, 224//4))
        seg_mask = seg_mask.astype(np.float64)/255.0
        heatmap = 0.5 * seg_mask + (1 - 0.5) * heatmap
        object_channel = chong_imutils.get_object_box_channel(gt_bboxes[:-1],width,height,resolution=self.input_size).unsqueeze(0)
        head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)
        head_box = gt_bboxes[-1]
        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform2(face)
        if self.training == 'test':
            return img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap,head_box, gtbox
        else:
            return img, face, location_channel,object_channel,head_channel ,head,gt_label,heatmap,head_box, gtbox


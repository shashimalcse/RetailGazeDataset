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

class RetailGaze(Dataset):
        def __init__(self, root_dir, mat_file, training='train', include_path=False, input_size=224, output_size=64, imshow = False, use_gtbox=False):
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
            # logging.info('%s contains %d images' % (self.mat_file, self.image_num))

        def __len__(self):
            return self.image_num

        def __getitem__(self, idx):
            gaze_inside = True
            data = self.data[idx]
            image_path = data['filename']
            image_path = os.path.join(self.root_dir, image_path)

            gaze = [float(data['gaze_cx'])/640, float(data['gaze_cy'])/480]
            gaze_x, gaze_y = gaze

            image_path = image_path.replace('\\', '/')
            img = Image.open(image_path)
            img = img.convert('RGB')
            width, height = img.size
            #Get bounding boxes and class labels as well as gt index for gazed object
            gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
            gt_labels = np.expand_dims(gt_labels, axis=0)
            if self.use_gtbox:
                gt_bboxes = np.copy(data['ann']['bboxes']) / [640, 480, 640, 480]
                gt_labels = np.copy(data['ann']['labels'])
            hbox = np.copy(data['ann']['hbox'])
            x_min, y_min, x_max, y_max = hbox
            head_x=((x_min+x_max)/2)
            head_y=((y_min+y_max)/2)
            eye = np.array([head_x, head_y])
            head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            mask_path = image_path.split('/')[:-1]
            mask_path = '/'.join(mask_path) + "/combined.png"
            mask = cv2.imread(mask_path,0)
            mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)
            mask_tensor = image_to_tensor(mask)
            object_channel = mask_tensor/255.0
            fov_path = image_path.split('/')
            fov_path[-1] = fov_path[-1].split('.')[0]
            fov_path = "".join(fov_path[-3:])
            fov = torch.load("/content/drive/MyDrive/RetailGaze/masks/"+mask_path)
            if self.imshow:
                img.save("img_aug.jpg")
                face.save('face_aug.jpg')

            if self.transform is not None:
                img = self.transform(img)
                face = self.transform2(face)

            # generate the heat map used for deconv prediction
            gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
            if self.training == 'test':  # aggregated heatmap
                gaze_heatmap = chong_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                            3,
                                                            type='Gaussian')

            else:
                # if gaze_inside:
                gaze_heatmap = chong_imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                    3,
                                                    type='Gaussian')


            if self.training == 'test':
                return img, face, head_channel,object_channel,fov, torch.from_numpy(eye),gaze_heatmap, image_path
            else:
                return img, face, head_channel,object_channel,fov,torch.from_numpy(eye),gaze_heatmap, image_path



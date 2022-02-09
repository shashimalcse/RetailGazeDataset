import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
# from models.gazenet import GazeNet

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


import pickle
from skimage import io
from dataloader import chong_imutils
from pytorchcvtcolor.image import image_to_tensor

def _get_transform(input_resolution):
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
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
            # eyess = np.array([eye[0],eye[1]]).astype(np.float)
            gaze_x, gaze_y = gaze

            image_path = image_path.replace('\\', '/')
            img = Image.open(image_path)
            img = img.convert('RGB')
            width, height = img.size
            #Get bounding boxes and class labels as well as gt index for gazed object
            gt_bboxes, gt_labels = np.zeros(1), np.zeros(1)
            gt_labels = np.expand_dims(gt_labels, axis=0)
            width, height = img.size
            hbox = np.copy(data['ann']['hbox'])
            x_min, y_min, x_max, y_max = hbox
            head_x=((x_min+x_max)/2)/640
            head_y=((y_min+y_max)/2)/480
            eye = np.array([head_x, head_y])
            eye_x, eye_y = eye
            k = 0.1
            x_min = (eye_x - 0.15) * width
            y_min = (eye_y - 0.15) * height
            x_max = (eye_x + 0.15) * width
            y_max = (eye_y + 0.15) * height
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0
            if x_max < 0:
                x_max = 0
            if y_max < 0:
                y_max = 0
            x_min -= k * abs(x_max - x_min)
            y_min -= k * abs(y_max - y_min)
            x_max += k * abs(x_max - x_min)
            y_max += k * abs(y_max - y_min)
            x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

            mask_path = image_path.split('/')[:-1]
            mask_path = '/'.join(mask_path) + "/combined.png"
            mask = cv2.imread(mask_path,0)
            mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)
            mask_tensor = image_to_tensor(mask)
            object_channel = mask_tensor/255.0

            head_channel = chong_imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)

            # Crop the face
            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            grid_size = 5
            gaze_label_size = 5
            v_x = [0, 1, -1, 0, 0]
            v_y = [0, 0, 0, -1, 1]


            shifted_grids = np.zeros((grid_size, gaze_label_size, gaze_label_size))
            for i in range(5):

                x_grid = int(np.floor( gaze_label_size * gaze_x + (v_x[i] * (1/ (grid_size * 3.0))) ) )
                y_grid = int(np.floor( gaze_label_size * gaze_y + (v_y[i] * (1/ (grid_size * 3.0))) ) )

                if x_grid < 0:
                    x_grid = 0
                elif x_grid > 4:
                    x_grid = 4
                if y_grid < 0:
                    y_grid = 0
                elif y_grid > 4:
                    y_grid = 4

                try:
                    shifted_grids[i][y_grid][x_grid] = 1
                except:
                    exit()

            shifted_grids = torch.from_numpy(shifted_grids).contiguous()

            shifted_grids = shifted_grids.view(1, 5, 25)
            gaze_final = np.ones(100)
            gaze_final *= -1
            gaze_final[0] = gaze_x
            gaze_final[1] = gaze_y
            eyes_loc_size = 13
            eyes_loc = np.zeros((eyes_loc_size, eyes_loc_size))
            eyes_loc[int(np.floor(eyes_loc_size * eye_y))][int(np.floor(eyes_loc_size * eye_x))] = 1

            eyes_loc = torch.from_numpy(eyes_loc).contiguous()
            if self.imshow:
                img.save("img_aug.jpg")

            if self.transform is not None:
                img = self.transform(img)
                face = self.transform(face)

            if self.training == 'test':
                return img, face, head_channel, object_channel,gaze_final,eye,gt_bboxes,gt_labels
            else:
                return img, face, head_channel, object_channel,eyes_loc, image_path, gaze_inside , shifted_grids, gaze_final
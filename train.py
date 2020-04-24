import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torchvision import transforms
import torchvision.datasets.folder as torch_folder
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec

# step 0: read train/test split
split = sio.loadmat(drive_path + "data/VOC2012/split.mat")
train_loc = split['train_loc']
print(train_loc.shape)
test_loc = split['test_loc']
print(test_loc.shape)
# Step 1: resize images
train_features = sio.loadmat(drive_path + "data/VOC2012/images_224_224_RGB_train.mat")['features']
print(train_features.shape)
test_features = sio.loadmat(drive_path + "data/VOC2012/images_224_224_RGB_test.mat")['features']
print(test_features.shape)
# Step 2: prepare seen classes, unseen classes and their vectors
class_info = sio.loadmat(drive_path + "data/VOC2012/classes_info.mat")
class_name = class_info['class_name']
seen_c = class_info['seen_c'][0]
unseen_c = class_info['unseen_c'][0]
class_vector = class_info['class_vector']
print(class_vector.shape)
# Step 3: read and resize labels
class_color = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],
               [192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],
               [128,64,0],[0,192,0],[128,192,0],[0,64,128]]
train_labels = sio.loadmat(drive_path + "data/VOC2012/seg_class_labels.mat")['train_labels']
train_labels_filtered = sio.loadmat(drive_path + "data/VOC2012/seg_class_labels.mat")['train_labels_filtered']
test_labels = sio.loadmat(drive_path + "data/VOC2012/seg_class_labels.mat")['test_labels']
print(train_labels.shape)
print(train_labels_filtered.shape)
print(test_labels.shape)

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from PIL import Image
from torchvision import transforms
import torchvision.datasets.folder as torch_folder
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torchvision.models as models
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
from torch.utils.checkpoint import checkpoint

drive_path = "./drive/My Drive/Colab Notebooks/Relation_Network/"

# step 0: read train.txt and val.txt
train_txt = drive_path + "data/VOC2012/train.txt"
train_loc = []
f = open(train_txt, 'r')
for line in f.readlines():
  train_loc.append(line.strip())
train_loc = np.array(train_loc)
f.close()

val_txt = drive_path + "data/VOC2012/val.txt"
test_loc = []
f = open(val_txt, 'r')
for line in f.readlines():
  test_loc.append(line.strip())
test_loc = np.array(test_loc)
f.close()
sio.savemat(drive_path + "data/VOC2012/split.mat", {'train_loc': train_loc, 'test_loc': test_loc})

# Step 1: preprocess images
def preprocess_batch_img(batch_path):
  batch_num = len(batch_path)
  input_batch = torch.zeros([batch_num,3,224,224], dtype=torch.float64)
  preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  for i in range(batch_num):
    path = batch_path[i].strip()
    print(path)
    input_image = torch_folder.pil_loader(path)
    input_tensor = preprocess(input_image)
    input_batch[i] = input_tensor
  return input_batch

image_root = drive_path + "data/VOC2012/JPEGImages/"

train_img_path = [image_root + i + ".jpg" for i in train_loc]
train_features = preprocess_batch_img(train_img_path)
train_features = train_features.numpy()
print(train_features.shape)
sio.savemat(drive_path + "data/VOC2012/images_224_224_RGB_train.mat",{'features':train_features})

test_img_path = [image_root + i + ".jpg" for i in test_loc]
test_features = preprocess_batch_img(test_img_path)
test_features = test_features.numpy()
print(test_features.shape)
sio.savemat(drive_path + "data/VOC2012/images_224_224_RGB_test.mat",{'features':test_features})

# Step 2: prepare seen classes, unseen classes and their vectors
# PASCAL VOC 2012 has 20 classes:
class_names = ["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car","Cat","Chair","Cow","Dining_table","Dog","Horse","Motorbike","Person","Potted_plant","Sheep","Sofa","Train","Tv_monitor"]
seen_c = range(15)
unseen_c = range(15,20)

word2vec_path = drive_path + "data/word2vec/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

word2vectors = []
for c_name in class_names:
  words = c_name.strip().split('_')
  vec = np.array([0] * 300, dtype='float64')
  for w in words:
    if w in model.vocab:
      vec += model[w]
  word2vectors.append(vec)
word2vectors = np.array(word2vectors)
print(word2vectors.shape)

# pre-trained fastText model with common crawl
import io
fastText_path = drive_path + "data/fastText/crawl-300d-2M.vec"
fin = io.open(fastText_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
lines = fin.readlines()
crawl_voc_list = {} # read word index first to avoid running out of RAM
for i in range(n):
  word = lines[i].split()[0]
  crawl_voc_list[word]=i

ftvectors = []
for c_name in class_names:
  words = c_name.strip().split('_')
  vec = np.array([0] * 300, dtype='float64')
  for w in words:
    if w in crawl_voc_list:
      index = crawl_voc_list[w]
      tokens = [float(num) for num in lines[index].split()[1:]]
      vec += tokens
  ftvectors.append(vec)
ftvectors = np.array(ftvectors)
print(ftvectors.shape)

sio.savemat(drive_path + "data/VOC2012/matfiles/classes_info.mat",{'class_names':class_names,'seen_c':seen_c,'unseen_c':unseen_c,'word2vectors':word2vectors,'ftvectors':ftvectors})

# Step 3: read and resize labels and save to mat
colors of each class above. [0,0,0] is background.
class_color = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],[128,128,128],[64,0,0],
               [192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128],[0,64,0],
               [128,64,0],[0,192,0],[128,192,0],[0,64,128]]

def preprocessLabel(image_index):
  print(image_index)
  import cv2
  label = cv2.imread(drive_path + "data/VOC2012/SegmentationClass/"+image_index+".png")
  label = cv2.resize(label, (224,224), interpolation=cv2.INTER_NEAREST)
  label = np.array(label)
  # background: 0; classes: 1-20; other pixels(segmentation borders): -1
  l_converted = np.zeros((224,224),dtype=int) -1 
  for i in range(len(class_color)):
    color = class_color[i]
    args = np.where((label == [color[2],color[1],color[0]]).all(-1)) # opencv read in BGR not RGB
    l_converted[args] = i
  return l_converted

train_labels = np.empty((train_loc.shape[0],1,224,224), dtype=int)
for i in range(train_loc.shape[0]):
  index = train_loc[i]
  train_labels[i,0,:,:] = preprocessLabel(image_index=index)
print(train_labels.shape)

test_labels = np.empty((test_loc.shape[0],1,224,224), dtype=int)
for i in range(test_loc.shape[0]):
  index = test_loc[i]
  test_labels[i,0,:,:] = preprocessLabel(image_index=index)
print(test_labels.shape)

sio.savemat(drive_path + "data/VOC2012/seg_class_labels.mat",{'train_labels':train_labels,'test_labels':test_labels})

background: 0; classes: 1-20; other pixels(segmentation borders): -1
train_labels = sio.loadmat(drive_path + "data/VOC2012/seg_class_labels.mat")['train_labels']
test_labels = sio.loadmat(drive_path + "data/VOC2012/seg_class_labels.mat")['test_labels']
print(train_labels.shape)
print(test_labels.shape)
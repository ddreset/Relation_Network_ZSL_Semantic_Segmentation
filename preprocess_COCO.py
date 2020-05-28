import scipy.io as sio
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
import torchvision.datasets.folder as torch_folder
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import cv2
import math
import io

drive_path = "./"

# Step 0: list train files and val files
train_files = os.listdir(drive_path + "data/COCO-Stuff/annotations/train2017")
train_loc = [name[:-4] for name in train_files if '.png' in name]
val_files = os.listdir(drive_path + "data/COCO-Stuff/annotations/val2017")
val_loc = [name[:-4] for name in val_files if '.png' in name]
sio.savemat(drive_path + "data/COCO-Stuff/matfiles/split.mat", {'train_loc': train_loc, 'val_loc': val_loc})
print("train_loc: ", len(train_loc))
print("val_loc: ", len(val_loc))

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
    input_image = torch_folder.pil_loader(path)
    input_tensor = preprocess(input_image)
    input_batch[i] = input_tensor
  return input_batch

image_root = drive_path + "data/COCO-Stuff/images/"

train_file_num = math.ceil(len(train_loc)/2000)
for i in range(train_file_num):
  train_img_path = [image_root + "train2017/" + i + ".jpg" for i in train_loc[i*2000:(i+1)*2000]]
  train_features = preprocess_batch_img(train_img_path)
  train_features = train_features.numpy()
  print("from ", i*2000, " to ", (i+1)*2000, " shape: ", train_features.shape)
  sio.savemat(drive_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_train_"+str(i)+".mat",{'features':train_features})

val_file_num = math.ceil(len(val_loc)/2000)
for i in range(val_file_num):
  val_img_path = [image_root + "val2017/" + i + ".jpg" for i in val_loc[i*2000:(i+1)*2000]]
  val_features = preprocess_batch_img(val_img_path)
  val_features = val_features.numpy()
  print("from ", i*2000, " to ", (i+1)*2000, " shape: ", val_features.shape)
  sio.savemat(drive_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_test_"+str(i)+".mat",{'features':val_features})

# Step 2: prepare seen classes, unseen classes and their vectors
label_txt = drive_path + "data/COCO-Stuff/labels.txt"
class_names = []
f = open(label_txt, 'r')
for line in f.readlines():
  name = line.split(":")[0].strip().replace("-"," ")
  class_names.append(name)
f.close()
if class_names[0] == "unlabeled":
  class_names = class_names[1:]
print("class_names ",len(class_names))
# unseen: cow, giraffe, suitcase, frisbee, skateboard, carrot, scissors, cardboard, clouds, grass, playing field, river, road, tree and wall-concrete
unseen_c = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
seen_c = [i for i in range(182) if i not in unseen_c]
print("unseen_c ",len(unseen_c))
print("seen_c ",len(seen_c))

# Step 2: prepare seen classes, unseen classes and their vectors
label_txt = drive_path + "data/COCO-Stuff/labels.txt"
class_names = []
f = open(label_txt, 'r')
for line in f.readlines():
  name = line.split(":")[1].strip().replace("-"," ")
  class_names.append(name)
f.close()
if class_names[0] == "unlabeled":
  class_names = class_names[1:]
print("class_names ",len(class_names))
# unseen: cow, giraffe, suitcase, frisbee, skateboard, carrot, scissors, cardboard, clouds, grass, playing field, river, road, tree and wall concrete
unseen_c = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
seen_c = [i for i in range(182) if i not in unseen_c]
print("unseen_c ",len(unseen_c))
print("seen_c ",len(seen_c))

word2vec_path = drive_path + "data/word2vec/GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

word2vectors = []
for c_name in class_names:
  words = c_name.strip().split(' ')
  vec = np.array([0] * 300, dtype='float64')
  for w in words:
    if w in model.vocab:
      vec += model[w]
  word2vectors.append(vec)
word2vectors = np.array(word2vectors)
print("word2vectors ", word2vectors.shape)

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
  words = c_name.strip().split(' ')
  vec = np.array([0] * 300, dtype='float64')
  for w in words:
    if w in crawl_voc_list:
      index = crawl_voc_list[w]
      tokens = [float(num) for num in lines[index].split()[1:]]
      vec += tokens
  ftvectors.append(vec)
ftvectors = np.array(ftvectors)
print("ftvectors ", ftvectors.shape)

sio.savemat(drive_path + "data/COCO-Stuff/matfiles/classes_info.mat",{'class_names':class_names,'seen_c':seen_c,'unseen_c':unseen_c,'word2vectors':word2vectors,'ftvectors':ftvectors})

# Step 3: read and resize labels and save to mat
# COCO labels are 3-channel indexed images 0-181 are labels, 255 is 'unlabelled' or void class
# let's convert unlabelled label to 0, and class labels to 1-182
def preprocessLabel(image_path):
  label = cv2.imread(image_path)
  label = cv2.resize(label, (224,224), interpolation=cv2.INTER_NEAREST)
  label = np.array(label[:,:,0])
  label = label + 1
  unlabelled_args = np.where(label == 256)
  label[unlabelled_args] = 0
  return label

train_labels = np.empty((train_loc.shape[0],1,224,224), dtype=int)
for i in range(train_loc.shape[0]):
  index = train_loc[i]
  train_labels[i,0,:,:] = preprocessLabel(drive_path + "data/COCO-Stuff/annotations/train2017/"+index+".png")
print("train_labels ", train_labels.shape)
train_file_num = math.ceil(len(train_loc)/6000)
for i in range(train_file_num):
  sio.savemat(drive_path + "data/COCO-Stuff/matfiles/labels_train_"+str(i)+".mat",{'labels':train_labels[i*6000:(i+1)*6000]})

test_labels = np.empty((val_loc.shape[0],1,224,224), dtype=int)
for i in range(val_loc.shape[0]):
  index = val_loc[i]
  test_labels[i,0,:,:] = preprocessLabel(drive_path + "data/COCO-Stuff/annotations/val2017/"+index+".png")
print("test_labels", test_labels.shape)
sio.savemat(drive_path + "data/COCO-Stuff/matfiles/labels_test.mat",{'labels':test_labels})
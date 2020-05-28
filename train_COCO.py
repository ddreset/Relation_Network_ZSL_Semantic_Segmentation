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
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import argparse
import models
from iou import IoU_per_class

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Zero Shot Semantic Segmentation")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-test-b","--test_batch_size",type = int, default = 32)
parser.add_argument("-label-s","--label_size",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 100000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-4)
parser.add_argument("-lm","--load_model",type=str2bool, default=False, help="load pkl files before training")
parser.add_argument("-rn-m","--relation_model_file_name",type=str)
parser.add_argument("-r","--root_path",type=str, default="./")
parser.add_argument("-acc","--last_accuracy",type=float, default=0.0)
parser.add_argument("-H","--last_H",type=float, default=0.0)
parser.add_argument("-v","--vec",type=int, default=0, help="choose from: 0:word2vec; 1:fastText; 2: word2vec::fastText")
parser.add_argument("-i","--img_model",type=int, default=0, help="choose from: 0:U-Net(VGG16); 1:DeepLab(VGG16)")
parser.add_argument("-g-l","--gpu_list",type=str,help="gpu list for parallel computing e.g. 1,2,3")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.test_batch_size
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
LOAD_MODEL = args.load_model
relation_model_path = args.root_path + "models/" + args.relation_model_file_name + ".pkl"
best_zsl_model_path = args.root_path + "models/" + args.relation_model_file_name + "_best_zsl.pkl"
best_gzsl_model_path = args.root_path + "models/" + args.relation_model_file_name + "_best_gzsl.pkl"
VEC = args.vec
IMG_MODEL = args.img_model
gpu_list = list(map(int,args.gpu_list.split(",")))
GPU = gpu_list[0]

# Step 1: list train files and val files
train_loc = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/split.mat")['train_loc']
val_loc = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/split.mat")['val_loc']
print("train_loc: ", len(train_loc))
print("val_loc: ", len(val_loc))

# Step 2: resize images
# training images saved in 60 mat files
# test images saved in 3 mat files
test_features = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_test_0.mat")['features']
for i in range(1,3):
  test_features = np.concatenate((test_features, sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_test_"+str(i)+".mat")['features']))
print("test_features", test_features.shape)

# Step 3: prepare seen classes, unseen classes and their vectors
class_info = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/classes_info.mat")
class_names = class_info['class_names']
seen_c = class_info['seen_c'][0]
unseen_c = class_info['unseen_c'][0]
if VEC == 0: # word2vec
  word_vectors = class_info['word2vectors']  
  vec_d = 300
  print("load word2vec ", word_vectors.shape)
elif VEC == 1: # fastText
  word_vectors = class_info['ftvectors']  
  vec_d = 300
  print("load fastText ", word_vectors.shape)
elif VEC == 2: # word2vec::fastText
  word2vectors = class_info['word2vectors']  
  ftvectors = class_info['ftvectors']  
  word_vectors = np.concatenate((word2vectors,ftvectors),1)
  vec_d = 600
  print("load np.cat(word2vec, fastText) ", word_vectors.shape)

# Step 4: read and resize labels
test_labels = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/labels_test.mat")['labels']
print("test_labels ", test_labels.shape)

# do not load all training data together because it takes a long time
def loadTrainData(label_id): # 0-19
  train_features = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_train_"+str(label_id*3)+".mat")['features']
  train_features = np.concatenate((train_features, sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_train_"+str(label_id*3+1)+".mat")['features']))
  train_features = np.concatenate((train_features, sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_train_"+str(label_id*3+2)+".mat")['features']))
  train_features = torch.from_numpy(train_features)

  train_labels = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/labels_train_" + str(label_id) + ".mat")['labels']
  train_labels = torch.from_numpy(train_labels)

  train_data = TensorDataset(train_features, train_labels)
  return train_data

# Step 5: prepare dataset
train_data = loadTrainData(np.random.randint(20, size=1)[0])
test_features = torch.from_numpy(test_features)
test_labels = torch.from_numpy(test_labels)

# Step 6: define and init models
relation_network = models.initModel(vec_d, IMG_MODEL)
relation_network = torch.nn.DataParallel(relation_network, device_ids=gpu_list)
relation_network.cuda(GPU)

relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim,step_size=5000,gamma=0.5)

if LOAD_MODEL:
  print("load model: ",LOAD_MODEL)
  relation_network.load_state_dict(torch.load(relation_model_path))

last_accuracy = args.last_accuracy
last_H = args.last_H

# Step 7: episode training
for episode in range(EPISODE):
  relation_network.train()
  relation_network_scheduler.step(episode)

  train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
  batch_features, batch_labels = train_loader.__iter__().next()

  episode_classes = np.unique(batch_labels)
  if episode_classes.shape[0] > args.label_size:
    episode_classes = np.sort(np.random.choice(episode_classes,args.label_size))
  # episode_classes = np.unique(np.random.randint(182, size=args.label_size)+1)
  filtered = np.isin(episode_classes, seen_c+1)
  filtered = np.where(filtered)
  sample_features = torch.tensor([word_vectors[int(c-1)] for c in episode_classes[filtered]]).view(-1,vec_d).float().cuda(GPU) # -1*300
  class_num = sample_features.shape[0]
  sample_features = sample_features.repeat(BATCH_SIZE,1)

  batch_features = batch_features.cuda(GPU).float() # -1*3*224*224
  batch_features = batch_features.repeat(1,class_num,1,1).view(-1,3,224,224)

  relations = relation_network(batch_features,sample_features).view(BATCH_SIZE,class_num,224,224)
  
  for c in range(1,183): # ignore other classes
    if c not in episode_classes[filtered]:
      c_args = np.where(batch_labels == c)
      batch_labels[c_args] = -1

  select_args = torch.where(batch_labels != -1) # only selected labels left
  one_hot_index = batch_labels[select_args]

  for c_i in range(class_num): # re-order labels
    c = episode_classes[filtered][c_i]
    c_args = np.where(one_hot_index == c)
    one_hot_index[c_args] = c_i+1
  
  one_hot_labels = torch.zeros(one_hot_index.shape[0],class_num+1).scatter_(1,one_hot_index.view(-1,1).long(),1)
  one_hot_labels = one_hot_labels[:,1:]
  relations = relations[select_args[0],:, select_args[2], select_args[3]]

  bce = nn.BCELoss().cuda(GPU)
  loss = bce(relations, one_hot_labels.cuda(GPU))

  loss.backward()
  # update
  relation_network_optim.step()
  relation_network.zero_grad()

  if (episode+1)%100 == 0:
    torch.save(relation_network.state_dict(),relation_model_path)
    print("episode:",episode+1,"loss",loss.item()," models saved!")
  if (episode+1)%1000 == 0:
    zsl_acc_per_class = IoU_per_class(relation_network, test_features, test_labels, word_vectors, unseen_c+1, TEST_BATCH_SIZE, GPU, None, None)
    print(zsl_acc_per_class)
    zsl_acc_per_class = [acc for acc in zsl_acc_per_class if acc is not None]
    zsl_mIoU = sum(zsl_acc_per_class) / len(zsl_acc_per_class)
    print('zsl = %.4f' % (zsl_mIoU))
    if zsl_mIoU > last_accuracy:
      last_accuracy = zsl_mIoU
      torch.save(relation_network.state_dict(), best_zsl_model_path) 
      print("Last zsl accuracy updated! Best zsl model saved!")
    else:
      print("last zsl accuracy is ", last_accuracy)

  if (episode+1)%10000 == 0:
    train_data = loadTrainData(np.random.randint(20, size=1)[0])
    print("loaded random training dataset")
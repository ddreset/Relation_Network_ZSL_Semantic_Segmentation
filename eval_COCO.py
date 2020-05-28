import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
import argparse
import models
from collections import OrderedDict
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
parser.add_argument("-test-b","--test_batch_size",type = int, default = 8)
parser.add_argument("-rn-m","--relation_model_file_name",type=str)
parser.add_argument("-r","--root_path",type=str, default="./")
parser.add_argument("-v","--vec",type=int, default=0, help="choose from: 0:word2vec; 1:fastText; 2: word2vec::fastText")
parser.add_argument("-i","--img_model",type=int, default=0, help="choose from: 0:U-Net(VGG16); 1:DeepLab(VGG16)")
parser.add_argument("-g-l","--gpu_list",type=str,help="gpu list for parallel computing e.g. 1,2,3")
parser.add_argument("-c","--calibrate",type=float, default=1.0)
parser.add_argument("-zsl","--zsl",type=str2bool, default=True)
parser.add_argument("-gzsl","--gzsl",type=str2bool, default=True)
args = parser.parse_args()

TEST_BATCH_SIZE = args.test_batch_size
relation_model_path = args.root_path + "models/" + args.relation_model_file_name + ".pkl"
VEC = args.vec
IMG_MODEL = args.img_model
gpu_list = list(map(int,args.gpu_list.split(",")))
GPU = gpu_list[0]

# Step 1: load resized images
# test images saved in 3 mat files
test_features = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_test_0.mat")['features']
for i in range(1,3):
  test_features = np.concatenate((test_features, sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/images_224_224_RGB_test_"+str(i)+".mat")['features']))
test_features = torch.from_numpy(test_features)
print("test_features", test_features.shape)

# Step 2: load seen classes, unseen classes and word vectors
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

# Step 3: load resized labels
test_labels = sio.loadmat(args.root_path + "data/COCO-Stuff/matfiles/labels_test.mat")['labels']
test_labels = torch.from_numpy(test_labels)
print("test_labels ", test_labels.shape)

# Step 4: define and init models
relation_network = models.initModel(vec_d, IMG_MODEL)
relation_network = torch.nn.DataParallel(relation_network, device_ids=gpu_list)
relation_network.cuda(GPU)
device = "cuda:" + str(GPU)
relation_network.load_state_dict(torch.load(relation_model_path,map_location=device))
relation_network.eval()
print("model loaded")

if args.zsl:
  with torch.no_grad():
    zsl_acc_per_class = IoU_per_class(relation_network, test_features, test_labels, word_vectors, unseen_c+1, TEST_BATCH_SIZE, GPU, None, None)
  print(zsl_acc_per_class)
  zsl_acc_per_class = [acc for acc in zsl_acc_per_class if acc is not None]
  zsl_mIoU = sum(zsl_acc_per_class) / len(zsl_acc_per_class)
  print('zsl = %.4f' % (zsl_mIoU))

if args.gzsl:
  gzsl_label_space = np.array(range(1,183))
  with torch.no_grad():
    gzsl_acc_per_class = IoU_per_class(relation_network, test_features, test_labels, word_vectors, gzsl_label_space, TEST_BATCH_SIZE, GPU, seen_c, args.calibrate)
  print(gzsl_acc_per_class)
  gzsl_acc_unseen = [gzsl_acc_per_class[c] for c in unseen_c]
  gzsl_acc_unseen = [acc for acc in gzsl_acc_unseen if acc is not None]
  unseen_mIoU = sum(gzsl_acc_unseen) / len(gzsl_acc_unseen)
  gzsl_acc_seen = [gzsl_acc_per_class[c] for c in seen_c]
  gzsl_acc_seen = [acc for acc in gzsl_acc_seen if acc is not None]
  seen_mIoU = sum(gzsl_acc_seen) / len(gzsl_acc_seen)
  H = (2 * seen_mIoU * unseen_mIoU) / (seen_mIoU + unseen_mIoU)
  print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (seen_mIoU, unseen_mIoU, H))

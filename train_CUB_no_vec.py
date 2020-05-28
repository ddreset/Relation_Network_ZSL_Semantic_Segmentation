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
import torchvision.models as models
import argparse
import models

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Zero Shot Semantic Segmentation on CUB 2011")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-test-b","--test_batch_size",type = int, default = 8)
parser.add_argument("-e","--episode",type = int, default= 50000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-6)
parser.add_argument("-lm","--load_model",type=str2bool, default=False, help="load pkl files before training")
parser.add_argument("-rn-m","--relation_model_file_name",type=str)
parser.add_argument("-acc","--last_accuracy",type=float, default=0.0)
parser.add_argument("-H","--last_H",type=float, default=0.0)
parser.add_argument("-r","--root_path",type=str, default="./")
parser.add_argument("-i","--img_model",type=int, default=0, help="choose from: 0:VGG16; 1:DeepLab(VGG16)")
parser.add_argument("-g-l","--gpu_list",type=str,help="gpu list for parallel computing e.g. 1,2,3")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.test_batch_size
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
relation_model_path = args.root_path + "models/" + args.relation_model_file_name + ".pkl"
IMG_MODEL = args.img_model
gpu_list = list(map(int,args.gpu_list.split(",")))
GPU = gpu_list[0]

# Step 1: read basic information
# class name list
# classes_file = args.root_path + "data/CUB_200_2011/classes.txt"
# class_names = []
# f = open(classes_file, 'r')
# for line in f.readlines():
#   l = line.split()
#   class_names.append(l[1].strip()[4:])
# class_names = np.array(class_names)
# f.close()

# image - class list
# image_class_labels_file = args.root_path + "data/CUB_200_2011/image_class_labels.txt"
# image_class = []
# f = open(image_class_labels_file, 'r')
# for line in f.readlines():
#   l = line.split()
#   image_class.append(int(l[1])-1) # in cub files, index starts with 0
# image_class = np.array(image_class)
# f.close()
# image_class = np.array(image_class)

# class_image = []
# for image_id in range(len(image_class)):
#   class_id = image_class[image_id]
#   if len(class_image) <= class_id:
#     class_image.append([image_id])
#   else:
#     class_image[class_id].append(image_id)
# class_image = np.array(class_image)

# image path list
# images_file = args.root_path + "data/CUB_200_2011/images.txt"
# image_paths = [''] * image_class.shape[0]
# f = open(images_file, 'r')
# for line in f.readlines():
#   l = line.split()
#   image_id = int(l[0])-1
#   image_paths[image_id] = l[1]
# f.close()
# image_paths = np.array(image_paths)

# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/basic_info.mat", {'class_names': class_names, 'image_class': image_class, 'class_image': class_image, 'image_paths': image_paths})

basic_info = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/basic_info.mat")
class_names = basic_info['class_names']
image_class = basic_info['image_class'][0]
class_image = basic_info['class_image'][0]
image_paths = basic_info['image_paths']

# Step 2: read train/test split and attributes
original_att_splits = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/original_att_splits.mat")
trainval_loc = original_att_splits['trainval_loc'].squeeze() - 1
test_seen_loc = original_att_splits['test_seen_loc'].squeeze() - 1
test_unseen_loc = original_att_splits['test_unseen_loc'].squeeze() - 1

# Step 3: read word embeddings (skip)

# Step 4: read all images and preprocess them
image_root = args.root_path + "data/CUB_200_2011/images/"

def preprocess_batch_img(batch_path):
  batch_num = len(batch_path)
  input_batch = torch.zeros([batch_num,3,224,224], dtype=torch.float64)
  preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  for i in range(batch_num):
    print(i)
    path = batch_path[i].strip()
    input_image = torch_folder.pil_loader(path)
    input_tensor = preprocess(input_image)
    input_batch[i] = input_tensor
  return input_batch

# train_img_path = [image_root + image_paths[i] for i in trainval_loc]
# train_features = preprocess_batch_img(train_img_path)
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_1.mat", {'features': train_features[0:2000].numpy()})
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_2.mat", {'features': train_features[2000:4000].numpy()})
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_3.mat", {'features': train_features[4000:6000].numpy()})
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_4.mat", {'features': train_features[6000:].numpy()})

train_features = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_1.mat")['features']
train_features = np.concatenate((train_features, sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_2.mat")['features']))
train_features = np.concatenate((train_features, sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_3.mat")['features']))
train_features = np.concatenate((train_features, sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_train_4.mat")['features']))

# test_seen_img_path = [image_root + image_paths[i] for i in test_seen_loc]
# test_seen_features = preprocess_batch_img(test_seen_img_path)
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_seen.mat", {'features': test_seen_features.numpy()})

test_seen_features = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_seen.mat")['features']

# test_unseen_img_path = [image_root + image_paths[i] for i in test_unseen_loc]
# test_unseen_features = preprocess_batch_img(test_unseen_img_path)
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_unseen_1.mat", {'features': test_unseen_features[0:2000].numpy()})
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_unseen_2.mat", {'features': test_unseen_features[2000:].numpy()})

test_unseen_features = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_unseen_1.mat")['features']
test_unseen_features = np.concatenate((test_unseen_features, sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/images_224_224_RGB_test_unseen_2.mat")['features']))

# Step 5: read all lebls
# label_root = args.root_path + "data/CUB_200_2011/segmentations/"
# labels = []
# for path in image_paths:
#   path = path.strip()
#   read_l = Image.open(label_root + path[0:-3]+'png').convert('L')
#   read_l = read_l.resize((224,224))
#   nparray = np.array(read_l)
#   nparray[np.where(nparray > 0)] = 1 # CUB's segmentation consists of the object and the background
#   labels.append(nparray)

# labels = np.array(labels)
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/seg_labels.mat", {'labels': labels})

labels = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/seg_labels.mat")['labels']

# Step 6: prepare datasests
train_features = torch.from_numpy(train_features)
train_labels = labels[trainval_loc]
train_labels = torch.from_numpy(train_labels)

train_data = TensorDataset(train_features, train_labels)

test_seen_features = torch.from_numpy(test_seen_features)
test_seen_labels = labels[test_seen_loc]
test_seen_labels = torch.from_numpy(test_seen_labels)
test_seen_classes = np.array([ image_class[img_id] for img_id in test_seen_loc])
test_seen_classes = torch.from_numpy(test_seen_classes)

test_unseen_features = torch.from_numpy(test_unseen_features)
test_unseen_labels = labels[test_unseen_loc]
test_unseen_labels = torch.from_numpy(test_unseen_labels)
test_unseen_classes = np.array([ image_class[img_id] for img_id in test_unseen_loc])
test_unseen_classes = torch.from_numpy(test_unseen_classes)

# Step 7 & 8: load pre-trained image embedding module define and init models

# init models
relation_network = models.initModel(0, IMG_MODEL)

relation_network = torch.nn.DataParallel(relation_network, device_ids=gpu_list)
relation_network.cuda(GPU)

relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim,step_size=5000,gamma=0.5)

if args.load_model:
  relation_network.load_state_dict(torch.load(relation_model_path))

last_accuracy = 0.0
last_H = 0.0

# mIoU calculation
def mIoU_of_class(prediction, predict_label, target, target_label):
  target_args = torch.where(target == target_label)
  target_size = target_args[0].shape[0]
  if target_size == 0:
    return None
  else:
    intersection = prediction[target_args] == predict_label
    intersection = torch.sum(intersection)
    predict_args = torch.where(prediction == predict_label)
    predict_size = predict_args[0].shape[0]
    union = target_size + predict_size - intersection
    mIoU = intersection.float()/union.float()
    return mIoU

def mIoU_per_class(test_features, test_labels, test_classes, test_batch): 
  relation_network.eval()
  test_data = TensorDataset(test_features, test_labels, test_classes)
  test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
  test_size = test_features.shape[0]

  class_set = np.unique(test_classes)
  class_num = len(class_set)
  class_acc = [None] * (class_num+1)
  predict_total = None

  for batch_features, batch_labels, batch_classes in test_loader:
    batch_size = batch_features.shape[0]
    query_features = batch_features.cuda(GPU).float()
    relations = relation_network(query_features).view(batch_size,1,224,224)
    relations = relations // 0.5 # 0 - background; 1 - target class.

    if predict_total is None:
      predict_total = relations.cpu().detach()
    else:
      predict_total = torch.cat((predict_total,relations.cpu().detach()))
 
  mIoU = mIoU_of_class(predict_total.view(-1,224,224), 0, test_labels.view(-1,224,224), 0)
  if mIoU is not None:
    class_acc[0] = mIoU.item()

  for c_i in range(class_num):
    class_id = class_set[c_i]
    image_ids = torch.where(test_classes==class_id)
    mIoU = mIoU_of_class(predict_total[image_ids].view(-1,224,224), 1, test_labels[image_ids].view(-1,224,224), 1)
    if mIoU is not None:
      class_acc[c_i+1] = mIoU.item()

  return class_acc

seen_acc_per_class = mIoU_per_class(test_seen_features ,test_seen_labels, test_seen_classes, TEST_BATCH_SIZE)
print(seen_acc_per_class)
seen_acc_per_class = [acc for acc in seen_acc_per_class if acc is not None]
seen_mIoU = sum(seen_acc_per_class) / len(seen_acc_per_class)

unseen_acc_per_class = mIoU_per_class(test_unseen_features ,test_unseen_labels, test_unseen_classes, TEST_BATCH_SIZE)
print(unseen_acc_per_class)
unseen_acc_per_class = [acc for acc in unseen_acc_per_class if acc is not None]
unseen_mIoU = sum(unseen_acc_per_class) / len(unseen_acc_per_class)
last_accuracy = unseen_mIoU

H = (2 * seen_mIoU * unseen_mIoU) / (seen_mIoU + unseen_mIoU)
last_H = H
print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (seen_mIoU, unseen_mIoU, H))

# Step 7: episode training
for episode in range(EPISODE):
  relation_network.train()
  relation_network_scheduler.step(episode)

  train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
  batch_features, batch_labels = train_loader.__iter__().next()
  batch_features = batch_features.cuda(GPU).float()  # -1*3*224*224
  relations = relation_network(batch_features).view(-1,224,224) # B, C, H, W

  bce = nn.BCELoss().cuda(GPU)
  loss = bce(relations, batch_labels.view(-1,224,224).float().cuda(GPU))

  # update
  relation_network.zero_grad()
  loss.backward()
  relation_network_optim.step()

  if (episode+1)%100 == 0:
    print("episode:",episode+1,"loss",loss.item())

  if (episode+1)%2000 == 0:
    unseen_acc_per_class = mIoU_per_class(test_unseen_features ,test_unseen_labels, test_unseen_classes, TEST_BATCH_SIZE)
    unseen_acc_per_class = [acc for acc in unseen_acc_per_class if acc is not None]
    unseen_mIoU = sum(unseen_acc_per_class) / len(unseen_acc_per_class)
    print('unseen=%.4f' % (unseen_mIoU))

    if unseen_mIoU > last_accuracy:
      last_accuracy = unseen_mIoU
      torch.save(relation_network.state_dict(),relation_model_path)
      print("last ZSL accuracy increased. models saved.")

    else:
      seen_acc_per_class = mIoU_per_class(test_seen_features ,test_seen_labels, test_seen_classes, TEST_BATCH_SIZE)
      seen_acc_per_class = [acc for acc in seen_acc_per_class if acc is not None]
      seen_mIoU = sum(seen_acc_per_class) / len(seen_acc_per_class)
      H = (2 * seen_mIoU * unseen_mIoU) / (seen_mIoU + unseen_mIoU)
      print('seen=%.4f, h=%.4f' % (seen_mIoU, H))

      if H > last_H:
        last_H = H
        torch.save(relation_network.state_dict(),relation_model_path)
        print("last H increased. models saved.")

print('best zsl accuracy=%.4f, best H=%.4f' % (last_accuracy, last_H))

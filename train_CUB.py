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
parser.add_argument("-e","--episode",type = int, default= 100000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-4)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-lm","--load_model",type=str2bool, default=False, help="load pkl files before training")
parser.add_argument("-vec-m","--vec_model_file_name",type=str)
parser.add_argument("-rn-m","--relation_model_file_name",type=str)
parser.add_argument("-acc","--last_accuracy",type=float, default=0.0)
parser.add_argument("-H","--last_H",type=float, default=0.0)
parser.add_argument("-r","--root_path",type=str, default="./")
parser.add_argument("-v","--vec",type=int, default=0, help="choose from: 0:attributes; 1:word2vec; 2:fastText; 3: word2vec::fastText")
parser.add_argument("-i","--img_model",type=int, default=0, help="choose from: 0:vgg16")
args = parser.parse_args()

BATCH_SIZE = args.batch_size
TEST_BATCH_SIZE = args.test_batch_size
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu # Google Colab offers 1 GPU for free
vec_model_path = args.root_path + "models/" + args.vec_model_file_name + ".pkl"
relation_model_path = args.root_path + "models/" + args.relation_model_file_name + ".pkl"
VEC = args.vec
IMG_MODEL = args.img_model

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
class_name = basic_info['class_names']
image_class = basic_info['image_class'][0]
class_image = basic_info['class_image'][0]
image_paths = basic_info['image_paths']

# Step 2: read train/test split and attributes
original_att_splits = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/original_att_splits.mat")
trainval_loc = original_att_splits['trainval_loc'].squeeze() - 1
test_seen_loc = original_att_splits['test_seen_loc'].squeeze() - 1
test_unseen_loc = original_att_splits['test_unseen_loc'].squeeze() - 1

# Step 3: read word embeddings
# pre-trained word2vec model with google news
# from gensim.models.keyedvectors import KeyedVectors
# from gensim.models import word2vec
# word2vec_path = args.root_path + "data/word2vec/GoogleNews-vectors-negative300.bin"
# model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
# word2vectors = []
# for c_name in class_name:
#   c_name = class_name[4:]
#   vec = [0] * 300
#   for word in class_name:
#     if word in model.vocab:
#       vec += model[word]
#   word2vectors.append(vec)
# word2vectors = np.array(word2vectors)
# print(word2vectors.shape)
# sio.savemat(args.root_path + "data/CUB_200_2011/matfiles/word_vectors.mat", {'word2vectors': word2vectors})

if VEC == 0: # attributes
  word_vectors = original_att_splits['att'].T
  vec_d = 312
  print("load attributes")
elif VEC == 1: # word2vec
  word_vectors = sio.loadmat(args.root_path + "data/CUB_200_2011/matfiles/word_vectors.mat")['word2vectors']  
  vec_d = 300
  print("load word2vec")

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
trainval_classes = np.array([ image_class[img_id] for img_id in trainval_loc])
trainval_classes = torch.from_numpy(trainval_classes)

train_data = TensorDataset(train_features, train_labels, trainval_classes)
# train_data = TensorDataset(train_features[0:2000], train_labels[0:2000], trainval_classes[0:2000])

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
class VecNetwork(nn.Module):
    def __init__(self,input_channel,output_channel,output_size):
        super(VecNetwork, self).__init__()
        self.up = nn.Upsample(size=(output_size,output_size))
        self.conv = nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=3,padding=1)

    def forward(self,x):
        x = self.up(x)
        x = F.relu(self.conv(x))
        return x

# relation network
class RelationNetworkWithVGG16(nn.Module):
    def __init__(self,vgg16,class_num):
        super(RelationNetworkWithVGG16, self).__init__()
        self.vgg16 = vgg16
        self.up1 = nn.Upsample(size=(112,112))
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=3,padding=1)
        self.up2 = nn.Upsample(size=(224,224))
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=3,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=3,out_channels=class_num,kernel_size=1)

    def forward(self,imgs,attributes):
        pre_x_112 = self.vgg16[0:5](imgs) # 64*112*112
        pre_x_28 = self.vgg16[5:17](pre_x_112) # 256*28*28
        x = torch.cat((pre_x_28,attributes),1)
        y = self.up1(x)
        y = F.relu(self.conv1(y))
        y = torch.cat((y,pre_x_112),1)
        y = self.up2(y)
        y = F.relu(self.conv2(y))
        y = torch.sigmoid(self.conv3(y))
        return y

# init models

vec_network = VecNetwork(vec_d,256,28)
if IMG_MODEL == 0:
  vgg16 = models.vgg16(pretrained=True)
  print("load pre-trained vgg16")
  relation_network = RelationNetworkWithVGG16(vgg16.features[0:17],1)


vec_network = torch.nn.DataParallel(vec_network)
relation_network = torch.nn.DataParallel(relation_network)
vec_network.cuda(GPU)
relation_network.cuda(GPU)

vec_network_optim = torch.optim.Adam(vec_network.parameters(),lr=LEARNING_RATE,weight_decay=1e-5)
vec_network_scheduler = StepLR(vec_network_optim,step_size=5000,gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
relation_network_scheduler = StepLR(relation_network_optim,step_size=5000,gamma=0.5)

if args.load_model:
  vec_network.load_state_dict(torch.load(vec_model_path))
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
  test_data = TensorDataset(test_features, test_labels, test_classes)
  test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
  test_size = test_features.shape[0]

  class_set = np.unique(test_classes)
  class_num = len(class_set)
  class_acc = [None] * (class_num+1)
  predict_total = None

  for batch_features, batch_labels, batch_classes in test_loader:
    batch_size = batch_features.shape[0]
    support_features = torch.tensor([word_vectors[c] for c in batch_classes]).view(batch_size,-1,1,1).float()
    support_features = vec_network(support_features.cuda(GPU)) # -1*300*1*1 -> -1*256*28*28
    query_features = batch_features.cuda(GPU).float()
    relations = relation_network(query_features,support_features).view(batch_size,1,224,224)
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
  vec_network_scheduler.step(episode)
  relation_network_scheduler.step(episode)

  train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
  batch_features, batch_labels, batch_classes, = train_loader.__iter__().next()

  support_features = torch.tensor([word_vectors[c] for c in batch_classes]).view(BATCH_SIZE,-1,1,1).float()
  support_features = vec_network(support_features.cuda(GPU)) # attribute features 32*300*1 -> 32*256*28*28
  batch_features = batch_features.cuda(GPU).float()  # -1*3*224*224
  relations = relation_network(batch_features,support_features).view(-1,224,224) # B, C, H, W

  bce = nn.BCELoss().cuda(GPU)
  loss = bce(relations, batch_labels.view(-1,224,224).float().cuda(GPU))

  # update
  vec_network.zero_grad()
  relation_network.zero_grad()
  loss.backward()
  vec_network_optim.step()
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
      torch.save(vec_network.state_dict(),vec_model_path)
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
        torch.save(vec_network.state_dict(),vec_model_path)
        torch.save(relation_network.state_dict(),relation_model_path)
        print("last H increased. models saved.")

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class RelationNetworkWithVGG16(nn.Module):
    def __init__(self,vgg16,class_num,vec_d):
        super(RelationNetworkWithVGG16, self).__init__()
        self.vgg16 = vgg16
        for p in vgg16.parameters():
          p.requires_grad = False
        self.up1 = nn.Upsample(size=(14,14))
        self.conv1 = nn.Conv2d(in_channels=1536,out_channels=512,kernel_size=3,padding=1)
        self.up2 = nn.Upsample(size=(28,28))
        self.conv2_1 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=768,out_channels=256,kernel_size=3,padding=1)
        self.up3 = nn.Upsample(size=(56,56))
        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=384,out_channels=128,kernel_size=3,padding=1)
        self.up4 = nn.Upsample(size=(112,112))
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=192,out_channels=64,kernel_size=3,padding=1)
        self.up5 = nn.Upsample(size=(224,224))
        self.conv5_1 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=9,out_channels=3,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(in_channels=3,out_channels=class_num,kernel_size=1)

        self.fc1 = nn.Linear(vec_d,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,3)

    def forward(self,imgs,vecs):
        pre_x_112 = self.vgg16[0:5](imgs) # 64*112*112
        pre_x_56 = self.vgg16[5:10](pre_x_112) # 128*56*56
        pre_x_28 = self.vgg16[10:17](pre_x_56) # 256*28*28
        pre_x_14 = self.vgg16[17:24](pre_x_28) # 512*14*14
        pre_x_7 = self.vgg16[24:31](pre_x_14) # 512*7*7

        vecs_1 = F.relu(self.fc1(vecs),inplace=True) # 512
        vecs_2 = F.relu(self.fc2(vecs_1),inplace=True) # 512
        vecs_3 = F.relu(self.fc3(vecs_2),inplace=True) # 256
        vecs_4 = F.relu(self.fc4(vecs_3),inplace=True) # 128
        vecs_5 = F.relu(self.fc5(vecs_4),inplace=True) # 64
        vecs_6 = F.relu(self.fc6(vecs_5),inplace=True) # 3

        vecs_1 = vecs_1.unsqueeze(2).unsqueeze(3)
        vecs_2 = vecs_2.unsqueeze(2).unsqueeze(3)
        vecs_3 = vecs_3.unsqueeze(2).unsqueeze(3)
        vecs_4 = vecs_4.unsqueeze(2).unsqueeze(3)
        vecs_5 = vecs_5.unsqueeze(2).unsqueeze(3)
        vecs_6 = vecs_6.unsqueeze(2).unsqueeze(3)

        Y = torch.cat((pre_x_7,vecs_1.repeat(1,1,7,7)),1) # 1024*7*7
        y = self.up1(pre_x_7)
        y = torch.cat((y,pre_x_14,vecs_2.repeat(1,1,14,14)),1) # 1536*14*14
        y = F.relu(self.conv1(y),inplace=True) # 512*14*14

        y = self.up2(y) # 512*28*28
        y = F.relu(self.conv2_1(y),inplace=True) # 256*28*28
        y = torch.cat((y,pre_x_28,vecs_3.repeat(1,1,28,28)),1) # 768*28*28
        y = F.relu(self.conv2_2(y),inplace=True) # 256*28*28

        y = self.up3(y) # 256*56*56
        y = F.relu(self.conv3_1(y),inplace=True) # 128*56*56
        y = torch.cat((y,pre_x_56,vecs_4.repeat(1,1,56,56)),1)
        y = F.relu(self.conv3_2(y),inplace=True) # 128*56*56

        y = self.up4(y) # 128*112*112
        y = F.relu(self.conv4_1(y),inplace=True) # 64*112*112
        y = torch.cat((y,pre_x_112,vecs_5.repeat(1,1,112,112)),1)
        y = F.relu(self.conv4_2(y),inplace=True) # 64*112*112

        y = self.up5(y) # 64*224*224
        y = F.relu(self.conv5_1(y),inplace=True) # 3*224*224
        y = torch.cat((y,imgs,vecs_6.repeat(1,1,224,224)),1)
        y = F.relu(self.conv5_2(y),inplace=True) # 3*224*224

        y = torch.sigmoid(self.conv6(y))
        return y

class RelationNetworkWithVGG16NoVec(nn.Module):
    def __init__(self,vgg16,class_num):
        super(RelationNetworkWithVGG16NoVec, self).__init__()
        self.vgg16 = vgg16
        for p in vgg16.parameters():
          p.requires_grad = False
        self.up1 = nn.Upsample(size=(14,14))
        self.conv1 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,padding=1)
        self.up2 = nn.Upsample(size=(28,28))
        self.conv2_1 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.up3 = nn.Upsample(size=(56,56))
        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.up4 = nn.Upsample(size=(112,112))
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.up5 = nn.Upsample(size=(224,224))
        self.conv5_1 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=6,out_channels=3,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(in_channels=3,out_channels=class_num,kernel_size=1)

    def forward(self,imgs):
        pre_x_112 = self.vgg16[0:5](imgs) # 64*112*112
        pre_x_56 = self.vgg16[5:10](pre_x_112) # 128*56*56
        pre_x_28 = self.vgg16[10:17](pre_x_56) # 256*28*28
        pre_x_14 = self.vgg16[17:24](pre_x_28) # 512*14*14
        pre_x_7 = self.vgg16[24:31](pre_x_14) # 512*7*7

        y = self.up1(pre_x_7)
        y = torch.cat((y,pre_x_14),1) # 1024*14*14
        y = F.relu(self.conv1(y),inplace=True) # 512*14*14

        y = self.up2(y) # 512*28*28
        y = F.relu(self.conv2_1(y),inplace=True) # 256*28*28
        y = torch.cat((y,pre_x_28),1)
        y = F.relu(self.conv2_2(y),inplace=True) # 256*28*28

        y = self.up3(y) # 256*56*56
        y = F.relu(self.conv3_1(y),inplace=True) # 128*56*56
        y = torch.cat((y,pre_x_56),1)
        y = F.relu(self.conv3_2(y),inplace=True) # 128*56*56

        y = self.up4(y) # 128*112*112
        y = F.relu(self.conv4_1(y),inplace=True) # 64*112*112
        y = torch.cat((y,pre_x_112),1)
        y = F.relu(self.conv4_2(y),inplace=True) # 64*112*112

        y = self.up5(y) # 64*224*224
        y = F.relu(self.conv5_1(y),inplace=True) # 3*224*224
        y = torch.cat((y,imgs),1)
        y = F.relu(self.conv5_2(y),inplace=True) # 3*224*224

        y = torch.sigmoid(self.conv6(y))
        return y

class RelationNetworkWithVGG16AllConv(nn.Module):
    def __init__(self,vgg16,class_num,vec_d):
        super(RelationNetworkWithVGG16, self).__init__()
        self.vgg16 = vgg16
        for p in vgg16.parameters():
          p.requires_grad = False
        self.up1 = nn.Upsample(size=(14,14))
        self.conv1 = nn.Conv2d(in_channels=1536,out_channels=512,kernel_size=3,padding=1)
        self.up2 = nn.Upsample(size=(28,28))
        self.conv2_1 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=768,out_channels=256,kernel_size=3,padding=1)
        self.up3 = nn.Upsample(size=(56,56))
        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=384,out_channels=128,kernel_size=3,padding=1)
        self.up4 = nn.Upsample(size=(112,112))
        self.conv4_1 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=192,out_channels=64,kernel_size=3,padding=1)
        self.up5 = nn.Upsample(size=(224,224))
        self.conv5_1 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=9,out_channels=3,kernel_size=3,padding=1)
        self.conv6 = nn.Conv2d(in_channels=3,out_channels=class_num,kernel_size=1)

        self.vec_up1 = nn.Upsample(size=(7,7))
        self.vec_conv1 = nn.Conv2d(in_channels=vec_d,out_channels=512,kernel_size=3,padding=1)
        self.vec_up2 = nn.Upsample(size=(14,14))
        self.vec_conv2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.vec_up3 = nn.Upsample(size=(28,28))
        self.vec_conv3 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.vec_up4 = nn.Upsample(size=(56,56))
        self.vec_conv4 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.vec_up5 = nn.Upsample(size=(112,112))
        self.vec_conv5 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.vec_up6 = nn.Upsample(size=(224,224))
        self.vec_conv6 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=3,padding=1)


    def forward(self,imgs,vecs):
        pre_x_112 = self.vgg16[0:5](imgs) # 64*112*112
        pre_x_56 = self.vgg16[5:10](pre_x_112) # 128*56*56
        pre_x_28 = self.vgg16[10:17](pre_x_56) # 256*28*28
        pre_x_14 = self.vgg16[17:24](pre_x_28) # 512*14*14
        pre_x_7 = self.vgg16[24:31](pre_x_14) # 512*7*7

        vecs = vecs.unsqueeze(2).unsqueeze(3)
        vecs_1 = self.vec_up1(vecs)
        vecs_1 = self.vec_conv1(vecs_1)
        vecs_2 = self.vec_up2(vecs_1)
        vecs_2 = self.vec_conv2(vecs_2)
        vecs_3 = self.vec_up3(vecs_2)
        vecs_3 = self.vec_conv3(vecs_3)
        vecs_4 = self.vec_up4(vecs_3)
        vecs_4 = self.vec_conv4(vecs_4)
        vecs_5 = self.vec_up5(vecs_4)
        vecs_5 = self.vec_conv5(vecs_5)
        vecs_6 = self.vec_up6(vecs_5)
        vecs_6 = self.vec_conv6(vecs_6)

        Y = torch.cat((pre_x_7,vecs_1),1) # 1024*7*7
        y = self.up1(pre_x_7)
        y = torch.cat((y,pre_x_14,vecs_2),1) # 1536*14*14
        y = F.relu(self.conv1(y),inplace=True) # 512*14*14

        y = self.up2(y) # 512*28*28
        y = F.relu(self.conv2_1(y),inplace=True) # 256*28*28
        y = torch.cat((y,pre_x_28,vecs_3),1) # 768*28*28
        y = F.relu(self.conv2_2(y),inplace=True) # 256*28*28

        y = self.up3(y) # 256*56*56
        y = F.relu(self.conv3_1(y),inplace=True) # 128*56*56
        y = torch.cat((y,pre_x_56,vecs_4),1)
        y = F.relu(self.conv3_2(y),inplace=True) # 128*56*56

        y = self.up4(y) # 128*112*112
        y = F.relu(self.conv4_1(y),inplace=True) # 64*112*112
        y = torch.cat((y,pre_x_112,vecs_5),1)
        y = F.relu(self.conv4_2(y),inplace=True) # 64*112*112

        y = self.up5(y) # 64*224*224
        y = F.relu(self.conv5_1(y),inplace=True) # 3*224*224
        y = torch.cat((y,imgs,vecs_6),1)
        y = F.relu(self.conv5_2(y),inplace=True) # 3*224*224

        y = torch.sigmoid(self.conv6(y))
        return y

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class RelationNetworkWithDeepLab(nn.Module):
    def __init__(self,deeplab,class_num,vec_d):
        super(RelationNetworkWithDeepLab, self).__init__()
        self.deeplab = deeplab
        for p in deeplab.backbone.parameters():
          p.requires_grad = False
        self.conv1 = nn.Conv2d(in_channels=512,out_channels=64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=class_num,kernel_size=3,padding=1)

        self.fc1 = nn.Linear(vec_d,256)
        self.fc2 = nn.Linear(256,64)

    def forward(self,x,vecs):
        vecs_1 = F.relu(self.fc1(vecs),inplace=True)
        vecs_2 = F.relu(self.fc2(vecs_1),inplace=True)

        vecs_1 = vecs_1.unsqueeze(2).unsqueeze(3)
        vecs_2 = vecs_2.unsqueeze(2).unsqueeze(3)

        y = self.deeplab(x)['out']
        y = torch.cat((y, vecs_1.repeat(1,1,224,224)),1)
        y = F.relu(self.conv1(y))
        y = torch.cat((y, vecs_2.repeat(1,1,224,224)),1)
        y = self.conv2(y)
        y = torch.sigmoid(y)
        return y

def initModel(vec_d, model=0, allConv=False):
  if model == 0:
    vgg16 = models.vgg16(pretrained=True)
    print("load pre-trained vgg16")
    if vec_d > 0:
      if not allConv:
        relation_network = RelationNetworkWithVGG16(vgg16.features,1,vec_d)
      else:
        relation_network = RelationNetworkWithVGG16(vgg16.features,1,vec_d)
    else:
       relation_network = RelationNetworkWithVGG16AllConv(vgg16.features,1)
    return relation_network
  elif model == 1:
    # in not pre-trained deeplabv3_resnet101, its backbone is ImageNet pre-trained reset101
    deeplab = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=False)
    deeplab.classifier[4] = Identity()
    print("load pre-trained deepLab")
    relation_network = RelationNetworkWithDeepLab(deeplab,1,vec_d)
    return relation_network

if __name__ == "__main__":
  query = torch.ones(5,3,224,224)
  support = torch.ones(5,300)
  relation_network = initModel(300,0)
  result = relation_network(query, support)
  print(result.shape)
  relation_network = initModel(300,1)
  result = relation_network(query, support)
  print(result.shape)
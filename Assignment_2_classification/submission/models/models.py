import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import pretrainedmodels

class ResNet18(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet18,self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18_fc_stripped = nn.Sequential(*list(self.resnet18.children())[:-1])
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet18_fc_stripped.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(in_features=512, out_features=7)

    def forward(self, x):
        x = self.resnet18_fc_stripped(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet50,self).__init__()
        self.resnet50 = models.resnet50(pretrained=pretrained)
        self.resnet50_fc_stripped = nn.Sequential(*list(self.resnet50.children())[:-1])
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet50_fc_stripped.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(in_features=2048, out_features=7)

    def forward(self, x):
        x = self.resnet50_fc_stripped(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        return x

class DPN92(nn.Module):
    def __init__(self):
        super(DPN92,self).__init__()
        model=pretrainedmodels.dpn92(num_classes=1000, pretrained='imagenet+5k')
        self.model_fc_stripped = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(in_features=172032, out_features=7)
    def forward(self, x):
        x = self.model_fc_stripped(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x


if __name__ =='__main__':
    model=DPN92()
    print(model)
    data=torch.rand(2,3,256,256)
    print(data.shape)
    output=model(data)
    print(output.shape)


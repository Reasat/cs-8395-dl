import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class ResNet18(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet18,self).__init__()
        self.resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18_fc_stripped = nn.Sequential(*list(self.resnet18.children())[:-1])
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet18_fc_stripped.parameters():
                param.requires_grad = False
        self.fc1 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        x = self.resnet18_fc_stripped(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        return x
class ResNet18_conv_fc(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet18_conv_fc,self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet18_fc_stripped = nn.Sequential(*list(resnet18.children())[:-2])
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet18_fc_stripped.parameters():
                param.requires_grad = False
        self.conv_last = nn.Conv2d(512,1,kernel_size=(1,1),stride=(1,1))
        self.fc1 = nn.Linear(in_features=16*11, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.resnet18_fc_stripped(x)
        x = self.conv_last(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ResNet8_conv_fc(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet8_conv_fc,self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.resnet_partial = nn.Sequential(nn.Sequential(*list(resnet18.children())[:5],
                                                          list(resnet18.children())[5][0]))
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet_partial.parameters():
                param.requires_grad = False
        self.conv_last = nn.Conv2d(128,1,kernel_size=(1,1),stride=(1,1))
        self.fc1 = nn.Linear(in_features=62*41, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.resnet_partial(x)
        # print(x.shape)
        x = self.conv_last(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ResNet12_conv_fc(nn.Module):
    def __init__(self, pretrained,bottleneckFeatures=1):
        super(ResNet12_conv_fc,self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        # print(resnet18)
        self.resnet_partial = nn.Sequential(*list(resnet18.children())[:6], *list(list(resnet18.children())[6][0].children())[:-1])
        if bottleneckFeatures ==1:
            print('freezing feature extracting layers')
            for param in self.resnet_partial.parameters():
                param.requires_grad = False
        self.conv_last = nn.Conv2d(256,1,kernel_size=(1,1),stride=(1,1))
        self.fc1 = nn.Linear(in_features=31*21, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.resnet_partial(x)
        # print(x.shape)
        x = self.conv_last(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
if __name__ =='__main__':
    model=ResNet12_conv_fc(pretrained=False)
    print(model)
    data=torch.rand(2,3,490,326)
    print(data.shape)
    output=model(data)
    print(output.shape)


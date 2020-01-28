import os
from matplotlib import pyplot as plt
import argparse
from glob import glob
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import  Mobile_Dataset_RAM, reverse_transform
from loss import loss_l2
from albumentations import Compose, Normalize
import albumentations.pytorch as albu_torch
from torch.utils.data import DataLoader
from models import ResNet18_flat_conv
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser=argparse.ArgumentParser()
parser.add_argument('--filepath',required=True)
parser.add_argument('--filepath_model',default=r'D:\Data\cs-8395-dl\model\2020-01-27-20-33-41\2020-01-27-20-33-41_resnet12_best.pt')
args = parser.parse_args()
dir_img = os.path.join(args.dir_data,args.partition)
filepaths = glob(os.path.join(dir_img, '*.jpg'))
# print(len(filepaths))
flnames = [os.path.basename(path) for path in filepaths]

filepath_label = os.path.join(args.dir_data, 'labels', 'labels.txt')
with open(filepath_label, 'r') as f:
    label_data = f.readlines()

label_dict={}
for data in label_data:
    name, x, y = data.strip().split(' ')
    label_dict[name]=(float(x),float(y))

aug = Compose([
    # Resize(256,256),
    #            RandomRotate90(),
    Normalize(),
    albu_torch.ToTensorV2()
],
)

Dataset = Mobile_Dataset_RAM(dir_data=dir_img, files=flnames, label_dict=label_dict, transform=aug)
print('number of samples {}'.format(len(Dataset)))
loader=DataLoader(Dataset,batch_size=args.batchSize, shuffle=False)
if 'resnet18' in args.filepath_model:
    model = ResNet18_flat_conv(pretrained=False,bottleneckFeatures=False).to(device)

train_states=torch.load(args.filepath_model)
print('loading model from epoch {}, with criteria {}'.format(train_states['epoch'],train_states['model_save_criteria']))
model.load_state_dict(train_states['model_state_dict'])
model.eval()
running_loss = 0
with torch.no_grad():
    for i, sample in enumerate(loader):
        img = sample[0].to(device)
        target = sample[1].to(device)
        output = model(img)
        loss = loss_l2(target, output)
        print(target.squeeze(), output.squeeze(),loss.item())
        img_r = reverse_transform(img.cpu().squeeze())
        print(img_r.shape)
        plt.imshow(img_r)
        plt.plot(target.cpu().squeeze()[0] * img_r.shape[1], target.cpu().squeeze()[1]* img_r.shape[0], 'r*')
        plt.plot(output.cpu().squeeze()[0] * img_r.shape[1], output.cpu().squeeze()[1] * img_r.shape[0], 'b*')
        plt.show()
        running_loss += loss.item()
        mean_loss = running_loss / (i + 1)
print('evaluate >>>  mean_loss: {}'.format(mean_loss))

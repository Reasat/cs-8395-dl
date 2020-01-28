import os
from matplotlib import pyplot as plt
import argparse
from glob import glob
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import  Mobile_Dataset_RAM, reverse_transform
from albumentations import Compose, Normalize
import albumentations.pytorch as albu_torch
from torch.utils.data import DataLoader
from models import ResNet12_conv_fc
import torch
import torch.nn as nn
from skimage import io, transform
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser=argparse.ArgumentParser()
parser.add_argument('--filepath',required=True)
parser.add_argument('--filepath_model',default=r'..\model_weight\2020-01-27-20-33-41_resnet12_best.pt')
args = parser.parse_args()
dir_img = os.path.dirname(args.filepath)
flnames = os.path.basename(args.filepath)
aug = Compose([
    Normalize(),
    albu_torch.ToTensorV2()
],
)
Dataset = Mobile_Dataset_RAM(dir_data=dir_img, files=[flnames], label_dict=None, transform=aug)
print('number of samples {}'.format(len(Dataset)))
loader=DataLoader(Dataset,batch_size=1, shuffle=False)
model = ResNet12_conv_fc(pretrained=False,bottleneckFeatures=False).to(device)
res_last_conv = nn.Sequential(*list(model.children())[:-2])
train_states=torch.load(args.filepath_model)
print('loading model from epoch {}, with criteria {}'.format(train_states['epoch'],train_states['model_save_criteria']))
model.load_state_dict(train_states['model_state_dict'])
model.eval()
with torch.no_grad():
    for i, sample in enumerate(loader):
        img = sample[0].to(device)
        output = res_last_conv(img)
        img_r = reverse_transform(img.cpu().squeeze())
        am_np=output.squeeze().cpu().numpy()
        am_np_rz=transform.resize(
                am_np,
                img_r.shape[:2])
        x,y=np.unravel_index(am_np_rz.argmax(), am_np_rz.shape)
        print(x / img_r.shape[0], y / img_r.shape[1])
        print('row, column ==> {:.4f}, {:.4f}'.format(x / img_r.shape[0], y / img_r.shape[1]))
        plt.imshow(img_r)
        plt.imshow(
            am_np_rz,
            alpha=0.3
        )
        plt.plot(y,x,'ro')
        plt.show()


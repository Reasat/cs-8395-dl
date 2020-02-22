from glob import glob
import os
from albumentations import (
    Compose, Resize, Normalize, RandomBrightnessContrast, HorizontalFlip,
CenterCrop
)
import albumentations.pytorch as albu_torch
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import ISIC_Dataset
from logger import Logger
from loss import bceWithSoftmax
from torch.utils.data import DataLoader
from models import ResNet18, ResNet50, DPN92
import torch.optim as optim
import torch
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from metrics import get_acc,get_recall,conf_mat
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import pretty_plot_confusion_matrix
from pandas import DataFrame
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser=argparse.ArgumentParser()
parser.add_argument('--filepath', help='project directory', required=True)
parser.add_argument('--encoder',help='encoder',default='dpn92')
parser.add_argument('--batchSize', help='batch size', type=int, default=32)
parser.add_argument('--load_from', help='filepath to load model', default=r'D:\Data\cs-8395-dl\model\2020-02-10-22-06-20\2020-02-10-22-06-20_dpn92_best.pt')
parser.add_argument('--resize', type=int, default=256)

args=parser.parse_args()

# setting up directories

BATCH_SIZE=args.batchSize
dir_data_part = os.path.dirname(args.filepath)
files = os.path.basename(args.filepath).split('.')[0]

# Dataloader Parameters
aug =Compose([
    Resize(args.resize,args.resize),
    Normalize(),
    albu_torch.ToTensorV2()
    ])

Dataset_valid = ISIC_Dataset(dir_data=dir_data_part, files=[files], label_cat=[ 0 ], do_cc=True,transform=aug)
loader_valid=DataLoader(Dataset_valid,batch_size=BATCH_SIZE, shuffle=False)
# print('validation samples {}'.format(len(Dataset_valid)))
# Model
if args.encoder == 'resnet18':
    model = ResNet18(pretrained=False, bottleneckFeatures=0).to(device)
if args.encoder == 'resnet50':
    model = ResNet50(pretrained=False, bottleneckFeatures=0).to(device)
if args.encoder == 'dpn92':
    model = DPN92().to(device)
# print(model)

# print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
# print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])

model.eval()
with torch.no_grad():
    for sample in loader_valid:
        img = sample[0].to(device)
        target = sample[1].to(device)
        output = model(img)
        output=torch.softmax(output,dim=1).detach().cpu().numpy()
        print(output.argmax())









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
parser.add_argument('--dir_project', help='project directory', default=r'..')
parser.add_argument('--dir_lf', help='directory large fileIDs',default=r'D:\Data\cs-8395-dl')
parser.add_argument('--folderData', help='data directory', default='assignment2_data')
parser.add_argument('--partition', help='partition', default='test')
parser.add_argument('--encoder',help='encoder',default='resnet18')
parser.add_argument('--batchSize', help='batch size', type=int, default=32)
parser.add_argument('--epoch', help='epoch', type=int, default=400)
parser.add_argument('--load_from', help='filepath to load model')
parser.add_argument('--resize', type=int)

args=parser.parse_args()

# setting up directories
DIR_LF = args.dir_lf#r'D:\Data\cs-8395-dl'
dir_data = os.path.join(DIR_LF,args.folderData) #os.path.join(DIR_LF,'assignment1_data')

# get train filenames
if args.partition=='train':
    dir_data_part= os.path.join(dir_data, 'train')
    filepath_label = os.path.join(dir_data, 'labels', 'Train_labels.csv')
    df_labels = pd.read_csv(filepath_label)
    df_labels.set_index('image', inplace=True)
    files = df_labels.index.values
    labels_one_hot=[df_labels.loc[flname].values for flname in files]
    labels_cat = [np.argmax(label) for label in labels_one_hot]
# get test filenames
if args.partition=='test':
    dir_data_part = os.path.join(dir_data, 'test')
    filepath_label = os.path.join(dir_data, 'labels', 'Test_labels.csv')
    df_labels = pd.read_csv(filepath_label)
    df_labels.set_index('image', inplace=True)
    files = df_labels.index.values
    labels_one_hot=[df_labels.loc[flname].values for flname in files]
    labels_cat = [np.argmax(label) for label in labels_one_hot]


# Dataloader Parameters
aug =Compose([
    Resize(args.resize,args.resize),
    Normalize(),
    albu_torch.ToTensorV2()
    ])


BATCH_SIZE=args.batchSize
EPOCH=args.epoch

Dataset_valid = ISIC_Dataset(dir_data=dir_data_part, files=files, label_cat=labels_cat, transform=aug)
loader_valid=DataLoader(Dataset_valid,batch_size=BATCH_SIZE, shuffle=False)
print('validation samples {}'.format(len(Dataset_valid)))
# Model
if args.encoder == 'resnet18':
    model = ResNet18(pretrained=False, bottleneckFeatures=0).to(device)
if args.encoder == 'resnet50':
    model = ResNet50(pretrained=False, bottleneckFeatures=0).to(device)
if args.encoder == 'dpn92':
    model = DPN92().to(device)
# print(model)

print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])

model.eval()
output_all=torch.FloatTensor([])
target_all=torch.FloatTensor([])
with torch.no_grad():
    for sample in tqdm(loader_valid):
        img = sample[0].to(device)
        target = sample[1].to(device)
        output = model(img)
        output_all=torch.cat((output_all,output.float().cpu()),dim=0)
        target_all=torch.cat((target_all,target.float().cpu()),dim=0)

recall_macro = get_recall(target_all, output_all, average='macro')
recall_micro = get_recall(target_all, output_all, average='micro')
recall_per_cls = get_recall(target_all, output_all, average=None)
mean_acc=get_acc(target_all, output_all)
cmat = conf_mat(target_all,output_all)
print('recall_per_cls: ', ['{:.4f}'.format(k) for k in recall_per_cls])
print('recall_micro: {:.4f}'.format(recall_micro))
print('recall_macro: {:.4f}'.format(recall_macro))
print('mean_acc: {:.4f}'.format(mean_acc))
print(df_labels.columns.values)
print(cmat)
recall = np.diag(cmat) / np.sum(cmat, axis = 1)
precision = np.diag(cmat) / np.sum(cmat, axis = 0)
recall_mean=np.mean(recall)
precision_mean = np.mean(precision)
print('recall: {:.2f}, precision: {:.2f}'.format(recall_mean, precision_mean))
df_cm = DataFrame(cmat, index=df_labels.columns.values, columns=df_labels.columns.values)
pretty_plot_confusion_matrix(df_cm,pred_val_axis='x')








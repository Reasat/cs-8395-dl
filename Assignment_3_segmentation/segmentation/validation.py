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
from dataloader import Spleen_Dataset
from logger import Logger
from loss import bceWithSoftmax
from torch.utils.data import DataLoader
from models import AlbuNet
import torch.optim as optim
import torch
from metrics import dice
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
from plotters import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser=argparse.ArgumentParser()
parser.add_argument('--dir_project', help='project directory', default=r'..')
parser.add_argument('--folderPartition', help='partition directory', default='train_test_org')
parser.add_argument('--dir_lf', help='directory large fileIDs',default=r'D:\Data\cs-8395-dl')
parser.add_argument('--path_kfold', help='kfold', default=r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\kfold_5.bin')
parser.add_argument('--folderData', help='data directory', default=r'assignment3\Training')
parser.add_argument('--encoder',help='encoder',default='resnet34')
parser.add_argument('--batchSize', help='batch size', type=int, default=8)
parser.add_argument('--load_from', help='filepath to load model',
                    default=r'D:\Data\cs-8395-dl\model\2020-03-22-06-05-03\2020-03-22-06-05-03_resnet34_fold-1_best.pt')

args=parser.parse_args()

# setting up directories
kf = pickle.load(open(args.path_kfold,'rb'))
BATCH_SIZE=args.batchSize

dir_data = os.path.join(args.dir_lf,args.folderData) #os.path.join(DIR_LF,'assignment1_data')
with open(os.path.join(args.dir_project,'partition',args.folderPartition,'Training.txt')) as  f:
    file_ids_all = [id.strip() for id in f.readlines()]

file_ids = np.array(file_ids_all)[list(kf.split(file_ids_all))[0][1]]
# file_ids = file_ids[]
print('patients {}'.format(file_ids))
Dataset = Spleen_Dataset(
            dir_data=dir_data,
            fileIDs=file_ids,
            axis='axial',
            # extract_spleen=True
        )
file_id_ind = range(len(Dataset))
# Dataloader Parameters
model = AlbuNet(pretrained=True, is_deconv=True).to(device)
# print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
# print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])
dice_coef = []
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats=False
with torch.no_grad():
    for ind in tqdm(file_id_ind):
        sample = Dataset[ind]
        output_3d = []
        mask_3d = []
        batch_start_range = range(0, sample[0].shape[-3], BATCH_SIZE)
        batch_start_range = np.array(list(batch_start_range))

        for i_b, batch_start in tqdm(enumerate(batch_start_range)):
            image_batch = sample[0].squeeze()[batch_start:batch_start + BATCH_SIZE][:, :]
            image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)
            image_batch = image_batch.float().to(device)

            mask_batch = sample[1].squeeze()[batch_start:batch_start + BATCH_SIZE][:, :]
            mask_batch = mask_batch.unsqueeze(1)
            mask_batch = mask_batch.float().to(device)

            output_batch = model(image_batch)

            # plot
            image_3d_rec,mask_3d_rec = recon_img_mask(image_batch, mask_batch)
            _, output_3d_rec = recon_img_mask(image_batch,output_batch)
            out_temp = torch.sigmoid(torch.tensor(output_3d_rec)).numpy()
            # output_3d_rec = (out_temp>out_temp.max()*.2 )*1
            # plot_outline(image_3d_rec, mask_3d_rec, output_3d_rec,
            #              range(0,output_3d_rec.shape[0],10))
            # plot_outline_heatmap(image_3d_rec, mask_3d_rec, out_temp,
            #              range(0,mask_3d_rec.shape[0],10))

            output_3d.append(output_batch.squeeze(1).detach().cpu().numpy())
            mask_3d.append(mask_batch.squeeze(1).detach().cpu().numpy())
        # print(len(output_3d), output_3d[0].shape, output_3d[11].shape)
        output_3d = np.concatenate(np.array(output_3d), axis=0 )
        output_3d_sig = torch.sigmoid(torch.tensor(output_3d)).numpy()
        mask_3d = np.concatenate(np.array(mask_3d), axis=0)

        dice_coef.append(dice((output_3d_sig>output_3d_sig.max()*.2)*1,mask_3d))
for f,d in zip(file_ids,dice_coef):
    print('{}: {:.2f}'.format(f,d))
print('dice mean: {:.4f}, median: {:.4f}, std: {:.4f}'.format(
    np.mean(dice_coef),
    np.median(dice_coef),
    np.std(dice_coef)
))







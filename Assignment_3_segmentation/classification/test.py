from glob import glob
import os
from albumentations import (
    Compose, Resize, Normalize, RandomBrightnessContrast, HorizontalFlip,
    CenterCrop
)
import albumentations.pytorch as albu_torch
import sys

sys.path.insert(1, r'..\utility')
sys.path.insert(1, r'..\models')
from dataloader import Spleen_Dataset
from logger import Logger
from loss import bceWithSoftmax
from torch.utils.data import DataLoader
from models import DPN68
import torch.optim as optim
import torch
import torch.nn as nn
from metrics import dice
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from metrics import get_acc, get_recall, conf_mat
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics import pretty_plot_confusion_matrix
from pandas import DataFrame
from plotters import plot_outline, recon_img_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# parser=argparse.ArgumentParser()
# parser.add_argument('--dir_project', help='project directory', default=r'..')
# parser.add_argument('--folderPartition', help='partition directory', default='train_test_org')
# parser.add_argument('--dir_lf', help='directory large fileIDs',default=r'D:\Data\cs-8395-dl')
# parser.add_argument('--path_kfold', help='kfold', default=r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\kfold_5.bin')
# parser.add_argument('--folderData', help='data directory', default=r'assignment3\Training')
# parser.add_argument('--encoder',help='encoder',default='resnet34')
# parser.add_argument('--batchSize', help='batch size', type=int, default=8)
# parser.add_argument('--load_from', help='filepath to load model',
#                     default=r'D:\Data\cs-8395-dl\model\2020-03-20-07-53-39\2020-03-20-07-53-39_1_fold-resnet34_best')

# args=parser.parse_args()
class Args():
    def __init__(self):
        self.dir_project = r'..'
        self.dir_lf = r'D:\Data\cs-8395-dl'
        self.path_kfold = r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\kfold_5.bin'
        self.folderPartition = 'train_test_org'
        self.folderData = r'assignment3\Testing'
        self.batchSize = 12
        self.axis = 'axial'
        self.lossWeight = [0.3, 0.7]
        self.load_from = r'D:/Data/cs-8395-dl/model/2020-03-23-07-39-40/2020-03-23-07-39-40_dpn68_fold-1_best.pt'
        self.dir_output = r'D:/Data/cs-8395-dl/output/assignment3/axial_spleen_prob'


args = Args()
# setting up directories
# kf = pickle.load(open(args.path_kfold,'rb'))
# ind_train, ind_valid = list(kf.split(file_ids))[0]
# file_ids_train = np.array(file_ids)[ind_train]
with open(os.path.join(args.dir_project, 'partition', args.folderPartition, 'Testing.txt')) as  f:
    file_ids = [id.strip() for id in f.readlines()]
file_ids_valid = file_ids
BATCH_SIZE = args.batchSize

dir_data = os.path.join(args.dir_lf, args.folderData)  # os.path.join(DIR_LF,'assignment1_data')
with open(os.path.join(args.dir_project, 'partition', args.folderPartition, 'Training.txt')) as  f:
    file_ids_all = [id.strip() for id in f.readlines()]

# Dataloader Parameters
model = DPN68().to(device)
print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])
Dataset_test = Spleen_Dataset(
    dir_data=dir_data,
    fileIDs=file_ids_valid,
    #         transform='organ_mask',
    axis=args.axis,
    no_label=True
    # extract_spleen=args.extract_spleen
    # to_ram=True

)
compute_loss = bceWithSoftmax(weights=args.lossWeight)

model.eval()
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = False
idx = range(len(Dataset_test))
with torch.no_grad():
    for ind in tqdm(idx):  # each patient
        sample = Dataset_test[ind]
        batch_start_range = range(0, sample.shape[0], BATCH_SIZE)
        batch_start_range = np.array(list(batch_start_range))
        spleen_prob = []
        for i_b, batch_start in tqdm(enumerate(batch_start_range)):  # each batch
            # sample[0] dim [147, 512, 512]
            image_batch = sample.squeeze()[batch_start:batch_start + BATCH_SIZE, :, :]

            image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)
            image_batch = image_batch.float().to(device)

            output = model(image_batch)
            spleen_prob.append(torch.softmax(output, dim=1).detach().cpu().numpy())
        #             print(target_batch)
        #             print(torch.softmax(output,dim=1))
        #             print(loss)

        #             # plot
        #             image_3d_rec, mask_3d_rec = recon_img_mask(image_batch, mask_batch)
        #             plot_outline(image_3d_rec, mask_3d_rec, torch.softmax(output,dim=1),
        #                          range(0, mask_3d_rec.shape[0], 1))
        #         break
        spleen_prob = np.concatenate(np.array(spleen_prob), axis=0)
        np.save(os.path.join(args.dir_output, '{}.npy'.format(file_ids_valid[ind])), spleen_prob)


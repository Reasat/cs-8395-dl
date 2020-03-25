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
import nibabel as nib
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
parser.add_argument('--dir_output',
                    default=r'D:\Data\cs-8395-dl\output\assignment3\segmentation_cc_thresh_axial_prob_no_mask_organ')

args=parser.parse_args()

# setting up directories
BATCH_SIZE=args.batchSize

dir_data = os.path.join(args.dir_lf,args.folderData) #os.path.join(DIR_LF,'assignment1_data')
with open(os.path.join(args.dir_project,'partition',args.folderPartition,'Training.txt')) as  f:
    file_ids_all = [id.strip() for id in f.readlines()]
kf = pickle.load(open(args.path_kfold,'rb'))
ind_train, ind_valid = list(kf.split(file_ids_all))[0]
file_ids = np.array(file_ids_all)[ind_valid]
# file_ids = file_ids[]
print('patients {}'.format(file_ids))
Dataset = Spleen_Dataset(
            dir_data=dir_data,
            fileIDs=file_ids,
            axis='axial',
            no_label=True
            # extract_spleen=True
        )
file_id_ind = range(len(Dataset))
# Dataloader Parameters
model = AlbuNet(pretrained=True, is_deconv=True).to(device)
# print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
# print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])
del train_states

dice_coef = []
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats=False

output_seg_bin = []
with torch.no_grad():
    for ind in tqdm(file_id_ind):
        sample = Dataset[ind]
        output_3d = []
        batch_start_range = range(0, sample.shape[-3], BATCH_SIZE)
        batch_start_range = np.array(list(batch_start_range))

        for i_b, batch_start in tqdm(enumerate(batch_start_range)):
            image_batch = sample.squeeze()[batch_start:batch_start + BATCH_SIZE][:, :]
            image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)
            image_batch = image_batch.float().to(device)

            output_batch = model(image_batch)

            # plot
            # image_3d_rec,mask_3d_rec = recon_img_mask(image_batch, mask_batch)
            # _, output_3d_rec = recon_img_mask(image_batch,output_batch)
            # out_temp = torch.sigmoid(torch.tensor(output_3d_rec)).numpy()
            # output_3d_rec = (out_temp>out_temp.max()*.2 )*1
            # plot_outline(image_3d_rec, mask_3d_rec, output_3d_rec,
            #              range(0,output_3d_rec.shape[0],10))
            # plot_outline_heatmap(image_3d_rec, mask_3d_rec, out_temp,
            #              range(0,mask_3d_rec.shape[0],10))

            output_3d.append(output_batch.squeeze(1).detach().cpu().numpy())
        # print(len(output_3d), output_3d[0].shape, output_3d[11].shape)
        output_3d = np.concatenate(np.array(output_3d), axis=0 )
        output_3d_sig = torch.sigmoid(torch.tensor(output_3d)).numpy()
        output_seg_bin.append((output_3d_sig>output_3d_sig.max()/2*1).transpose(1,2,0))
        # output_3d_sig_nib = output_3d_sig.transpose(1,2,0)
        # np.save(os.path.join(args.dir_output, 'label{}.npy'.format(file_ids[ind])), output_3d_sig.transpose(1,2,0))

del model

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
        self.folderData = r'assignment3\Training'
        self.batchSize = 12
        self.axis = 'axial'
        self.lossWeight = [0.3, 0.7]
        self.load_from = r'D:/Data/cs-8395-dl/model/2020-03-23-03-22-40/2020-03-23-03-22-40_dpn68_fold-1_best.pt'
        self.dir_output = r'D:\Data\cs-8395-dl\output\assignment3\segmentation_cc_thresh_axial_prob_no_mask_organ'


args = Args()
# setting up directories
# kf = pickle.load(open(args.path_kfold,'rb'))
# ind_train, ind_valid = list(kf.split(file_ids))[0]
# file_ids_train = np.array(file_ids)[ind_train]
BATCH_SIZE = args.batchSize

dir_data = os.path.join(args.dir_lf, args.folderData)  # os.path.join(DIR_LF,'assignment1_data')

# Dataloader Parameters
model = DPN68().to(device)
print('loading model from {}'.format(args.load_from))
train_states = torch.load(args.load_from)
print('loading model from epoch ', train_states['epoch'])
model.load_state_dict(train_states['model_state_dict'])
del train_states
Dataset = Spleen_Dataset(
    dir_data=dir_data,
    fileIDs=file_ids,
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
idx = range(len(Dataset))

spleen_prob_all = []
with torch.no_grad():
    for ind in tqdm(idx):  # each patient
        sample = Dataset[ind]
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
        spleen_prob_all.append(spleen_prob)
        # np.save(os.path.join(args.dir_output, '{}.npy'.format(file_ids_valid[ind])), spleen_prob)

del model

import cc3d
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.pyplot as plt
from plotly import graph_objs as go
import nibabel as nib
import os
from glob import glob
from tqdm import tqdm
from dataloader import *

for output, axial_prob, file_id in zip(output_seg_bin,spleen_prob_all, file_ids):
    # np.random.seed(42)
    labels_out = cc3d.connected_components(output,
                                           connectivity=6,
                                           out_dtype=np.uint16)

    # You can extract individual components like so:
    N = np.max(labels_out)
    criteria = 0
    segid_spleen = 0
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        vol = (extracted_image != 0).sum() / (extracted_image.size)
        _, _,z = extract_spleen_coord(extracted_image, label=segid)
        spleen_prob = np.median(axial_prob[z[0]:z[1], 1])
        metric = vol * spleen_prob
        print('segid: {}, Volume ratio: {:.6f}, median prob {:.2f}, vol*prob {:.6f}'.format(segid, vol, spleen_prob,
                                                                                            metric))
        if metric > criteria:
            print('updating criteria {:.6f}-->{:.6f} for seg_id {}'.format(criteria, metric, segid))
            criteria = metric
            segid_spleen = segid

    spleen_comp = np.ones_like(labels_out) * (labels_out == segid_spleen)
    path_save = os.path.join(args.dir_output,'label{}.nii.gz'.format(file_id))
    path_img = r'D:\Data\cs-8395-dl\assignment3\Training\img\img{}.nii.gz'.format(file_id)
    img_nib = nib.load(path_img)
    label_spleen = nib.Nifti1Image(spleen_comp, img_nib.affine, img_nib.header)
    nib.save(label_spleen, path_save)
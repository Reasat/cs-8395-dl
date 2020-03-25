from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
from skimage import io
import torch
from plotters import subplot_img_mask
from glob import glob
from skimage import io
from scipy.ndimage import gaussian_filter
from albumentations import (
Resize,HorizontalFlip,
    Compose,
    Normalize,
RandomBrightnessContrast,
CenterCrop,
)
import albumentations.pytorch as albu_torch
import pandas as pd
from matplotlib import pyplot as plt
import nibabel as nib
import sys

def randomize_batch(image_batch,mask_batch):
    # randomize batch samples
    image_batch_temp = image_batch
    mask_batch_temp = mask_batch
    batch_indx = np.arange(image_batch.shape[0])
    np.random.shuffle(batch_indx)
    for i_rand, i_org in zip(batch_indx,np.arange(image_batch.shape[0])):
        image_batch[i_org] = image_batch_temp[i_rand]
        mask_batch[i_org] = mask_batch_temp[i_rand]
    return image_batch,mask_batch

def reverse_transform(img_t,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_r = np.array(img_t)
    img_r = img_r.transpose([1,2,0])
    img_r = img_r*std+mean
    img_r *=255
    img_r=img_r.astype(np.uint8)
    img_r = np.squeeze(img_r)
    return img_r


def range_finder(label_np, label, offset=1):
    start = None
    end = None
    for i in range(0, label_np.shape[0], offset):
        if np.sum(label_np[i:i + offset, :, :] == label):
            start = i
            break
    for i in range(start, label_np.shape[0], offset):
        if np.sum(label_np[i:i + offset, :, :] == label) == 0:
            end = i
            break
    return start, end


def extract_spleen_coord(label_np, label=1, offset=1):
    # starting edge
    coord_start = [0, 0, 0]
    # ending edge
    coord_end = [0, 0, 0]
    coord_start[0], coord_end[0] = range_finder(label_np, label, offset)
    coord_start[1], coord_end[1] = range_finder(label_np.transpose(1, 0, 2), label, offset)
    coord_start[2], coord_end[2] = range_finder(label_np.transpose(2, 1, 0), label, offset)

    return [(x1, x2) for x1, x2 in zip(coord_start, coord_end)]


def extract_spleen(img, label):
    coords = extract_spleen_coord(label)
    # print(coords)
    return (img[coords[0][0]:coords[0][1]],
            label[coords[0][0]:coords[0][1]])


class Spleen_Dataset(Dataset):
    def __init__(self, dir_data, fileIDs, axis, to_ram = False,
                 transform=None, hide_spleen=False, extract_spleen=False,no_label=False):
        self.dir_data = dir_data
        self.transform = transform
        self.hide_spleen = hide_spleen
        self.fileIDs = fileIDs
        self.to_ram = to_ram
        self.extract_spleen = extract_spleen
        self.axis = axis
        self.image_3d_all=[]
        self.mask_3d_all=[]
        self.no_label =no_label
        if self.to_ram:
            print('loading images to RAM')
            for file in tqdm(self.fileIDs):
                # file = self.fileIDs[idx]
                image_3d=self.get_img(file)
                self.image_3d_all.append(image_3d)
                if self.no_label is False:
                    mask_3d = self.get_mask(file)
                    self.mask_3d_all.append(mask_3d)

    def get_img(self,file):
        path_img = os.path.join(self.dir_data, 'img', 'img{}.nii.gz'.format(file))
        image_3d = nib.load(path_img)
        image_3d = image_3d.get_fdata()
        if self.axis =='axial':
            image_3d = image_3d.transpose((2, 0, 1))
        if self.axis == 'sagittal':
            pass
        if self.axis == 'coronal':
            image_3d = image_3d.transpose((1, 0, 2))
        return image_3d

    def get_mask(self,file):
        path_mask = os.path.join(self.dir_data, 'label', 'label{}.nii.gz'.format(file))
        mask_3d = nib.load(path_mask)
        mask_3d = mask_3d.get_fdata()
        if self.axis =='axial':
            mask_3d = mask_3d.transpose((2, 0, 1))
        if self.axis == 'sagittal':
            pass
        if self.axis == 'coronal':
            mask_3d = mask_3d.transpose((1, 0, 2))

        # mask_3d[mask_3d != 1] = 0 # get only spleen
        return mask_3d

    def __len__(self):
        size = len(self.fileIDs)
        return size

    def __getitem__(self, idx):
        if self.to_ram:
            image_3d=self.image_3d_all[idx]
            if self.no_label is False:
                mask_3d = self.mask_3d_all[idx]
        else:
            image_3d = self.get_img(self.fileIDs[idx])
            if self.no_label is False:
                mask_3d = self.get_mask(self.fileIDs[idx])

        if self.extract_spleen:
            image_3d, mask_3d = extract_spleen(image_3d, mask_3d)
        # warning: check this filtering
        bg = -100
        image_3d[image_3d > 300] = bg
        image_3d[image_3d < -100] = bg
        # check the effect of this
        # normalizations properly

        # augmentation
        if self.transform is not None:
            if 'organ_mask' in self.transform:
                # print('using organ_mask augmentation')
                if self.hide_spleen == True:
                    organ_ids = np.arange(1,14)
                else:
                    organ_ids = np.arange(2, 14)
                organ_ids_subset = organ_ids
                # np.random.choice(organ_ids,7)
                for org_id in organ_ids_subset:
                    if np.random.randn() > 0.5: # hide organ
                        # print('coarse dropout')
                        inds_sample = np.arange(image_3d.shape[0])
                        subset = np.random.choice(inds_sample, len(inds_sample) // 2)
                        for ind in subset:
                            image_3d[ind][mask_3d[ind] == org_id] = bg
                            mask_3d[ind][mask_3d[ind] == org_id] = 0
        if self.no_label is False:
            mask_3d[mask_3d!=1]=0 # gt from only spleen
        image_3d = (image_3d - image_3d.min()) \
                       / (image_3d.max() - image_3d.min())
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        # print(image_3d.min(),image_3d.max())
        image_3d = (image_3d-0.485)/0.229
        # print(image_3d.min(), image_3d.max())
        if self.no_label is False:
            return torch.tensor(image_3d),torch.tensor(mask_3d)
        else:
            return torch.tensor(image_3d)

if __name__=='__main__':
    dir_data = r'D:\Data\cs-8395-dl\assignment3\Training'
    fileIDs = ['0001','0002']

    # aug = Compose([
    #     HorizontalFlip(),
    #     RandomBrightnessContrast(
    #         brightness_limit=[-0.2, 0.2],
    #         contrast_limit=[-0.2, 0.2],
    #     ),
    #     CenterCrop(256,256,p=0.5),
    #     Resize(256, 256),
    #     Normalize(),
    #     albu_torch.ToTensorV2()
    # ],
    # )

    BATCH_SIZE = 8
    dataset = Spleen_Dataset(dir_data=dir_data,
                           fileIDs=fileIDs,
                             # to_ram=True,
                           )
    for i_b, batch_start in tqdm(enumerate(batch_start_range)):
        # sample[0] dim [147, 512, 512], sample[1] [147, 512, 512]
        image_batch = sample[0][batch_start:batch_start + BATCH_SIZE, :, :]
        mask_batch = sample[1][batch_start:batch_start + BATCH_SIZE, :, :]

        if np.random.randn() > 0.5:
            inds_sample = np.arange(image_batch.shape[0])
            subset = np.random.choice(inds_sample, len(inds_sample) // 2)
            for ind in subset:
                image_batch[ind][mask_batch[ind] == 1] = 0
                mask_batch[ind][mask_batch[ind] == 1] = 0

        image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)

        image_batch = image_batch
        target_batch = ((torch.sum(mask_batch, dim=(1, 2)) > 0) * 1)
        target_batch = target_batch
        # print(target_batch)
        # augmentation

        # # plot
        # if epoch+1 == args.epoch and i_p == 0 and i_b == 0:
        #     image_3d_rec, mask_3d_rec = recon_img_mask(image_batch, mask_batch)
        #     _, output_3d_rec = recon_img_mask(image_batch, output)
        #     plot_heatmap(image_3d_rec, output_3d_rec,
        #                  range(0, output_3d_rec.shape[0], 2))
        #     output_3d_rec_bin = (output_3d_rec > output_3d_rec.max() / 2) * 1
        #     print('out_max',output_3d_rec_bin.max())
        #     plot_outline(image_3d_rec, mask_3d_rec, output_3d_rec_bin,
        #                  range(0, output_3d_rec_bin.shape[0], 2))





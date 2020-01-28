from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
import cv2
import torch
from glob import glob
from skimage import io
from scipy.ndimage import gaussian_filter
from albumentations import (
    Compose,
    Normalize,
)
import albumentations.pytorch as albu_torch

from matplotlib import pyplot as plt

def reverse_transform(img_t,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_r = np.array(img_t)
    img_r = img_r.transpose([1,2,0])
    img_r = img_r*std+mean
    img_r *=255
    img_r=img_r.astype(np.uint8)
    img_r = np.squeeze(img_r)
    return img_r

class Mobile_Dataset_RAM(Dataset):
    def __init__(self, dir_data, files, label_dict, transform=None):
        self.dir_data = dir_data
        self.transform = transform
        self.files = files
        self.image_all=[]
        self.label_dict=label_dict
        print('loading images to RAM')
        for file in tqdm(self.files):
            # file = self.files[idx]
            path_img = os.path.join(self.dir_data, file)
            image = cv2.imread(path_img)
            self.image_all.append(image)

    def __len__(self):
        size = len(self.files)
        return size

    def __getitem__(self, idx):
        image=self.image_all[idx]
        if 'test' in self.dir_data:
            target=[0.5,0.5]
        else:
            target=self.label_dict[self.files[idx]]
        # print(self.files[idx],image.shape)
        transformed=self.transform(image=image)
        img = transformed['image']
        return img,torch.tensor(target)

if __name__=='__main__':

    dir_data = r'D:\Data\cs-8395-dl\assignment1_data'
    dir_data_train = os.path.join(dir_data,'train')
    filepaths_train = glob(os.path.join(dir_data_train, '*.jpg'))
    filepaths_train_label = os.path.join(dir_data, 'labels', 'labels.txt')
    with open(filepaths_train_label, 'r') as f:
        label_data = f.readlines()
    label_dict = {}
    for data in label_data:
        name, x, y = data.strip().split(' ')
        label_dict[name] = (float(x), float(y))

    aug = Compose([
        # Resize(256,256),
        #            RandomRotate90(),
        Normalize(),
        albu_torch.ToTensorV2()
    ],
    )
    files = list(label_dict.keys())
    # Mobile_Dataset = Mobile_Dataset_RAM(dir_data=dir_data_train,files=files,label_dict=label_dict,transform=aug)
    # sample = Mobile_Dataset[0]
    # img = sample[0]
    # target = sample[1]
    # img = reverse_transform(img)
    # x,y = np.array(target)
    # plt.imshow(img)
    # plt.plot(x * img.shape[1], y * img.shape[0], 'r*')
    # plt.show()

    Mobile_Dataset = Mobile_Dataset_HM_RAM(dir_data=dir_data_train,files=files,label_dict=label_dict,transform=aug)
    sample = Mobile_Dataset[0]
    img = sample[0]
    target = sample[1].cpu().numpy()
    img = reverse_transform(img)
    plt.imshow(img)
    plt.imshow(target, alpha=0.3)
    plt.show()


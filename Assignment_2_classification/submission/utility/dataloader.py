from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
from skimage import io
import torch
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
from sampler import BalancedBatchSampler
import sys
sys.path.insert(1,r'..\preprocessing')
from color_constancy import ColorConstancy

def reverse_transform(img_t,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_r = np.array(img_t)
    img_r = img_r.transpose([1,2,0])
    img_r = img_r*std+mean
    img_r *=255
    img_r=img_r.astype(np.uint8)
    img_r = np.squeeze(img_r)
    return img_r

class ISIC_Dataset(Dataset):
    def __init__(self, dir_data, files, label_cat, to_ram = False, transform=None, do_cc=False):
        self.dir_data = dir_data
        self.transform = transform
        self.files = files
        self.to_ram = to_ram
        self.image_all=[]
        self.label_cat=label_cat
        self.do_cc = do_cc
        self.color_constancy = ColorConstancy(verbose=False, thresh_bg=None)
        if self.to_ram:
            print('loading images to RAM')
            for file in tqdm(self.files):
                # file = self.files[idx]
                path_img = os.path.join(self.dir_data, file+'.jpg')
                image = io.imread(path_img)
                if self.do_cc:
                    image=self.color_constancy.comp(image)
                # print(image.shape)
                self.image_all.append(image)

    def __len__(self):
        size = len(self.files)
        return size

    def __getitem__(self, idx):
        if self.to_ram:
            image=self.image_all[idx]
        else:
            # print(self.files[idx])
            path_img = os.path.join(self.dir_data, self.files[idx] + '.jpg')
            image = io.imread(path_img)
        target=self.label_cat[idx]
        # print(self.files[idx],image.shape)
        transformed=self.transform(image=image)
        img = transformed['image']
        return img,torch.tensor(target)

if __name__=='__main__':
    dir_data = r'D:\Data\cs-8395-dl\assignment2_data'
    dir_data_train = os.path.join(dir_data,'train')
    filepath_train_label = os.path.join(dir_data, 'labels','Train_labels.csv')
    filepath_test_label = os.path.join(dir_data, 'labels', 'Test_labels.csv')
    df_train = pd.read_csv(filepath_train_label)
    df_train.set_index('image',inplace=True)
    files = df_train.index.values
    labels_one_hot = [df_train.loc[flname].values for flname in files]
    labels_cat = [np.argmax(label) for label in labels_one_hot]

    aug = Compose([
        HorizontalFlip(),
        RandomBrightnessContrast(
            brightness_limit=[-0.2, 0.2],
            contrast_limit=[-0.2, 0.2],
        ),
        CenterCrop(256,256,p=0.5),
        Resize(256, 256),
        Normalize(),
        albu_torch.ToTensorV2()
    ],
    )


    dataset = ISIC_Dataset(dir_data=dir_data_train,
                           files=files[:100],
                           label_cat=labels_cat[:100],
                           transform=aug
                           )
    train_loader = DataLoader(dataset, sampler=BalancedBatchSampler(dataset, labels_cat,shuffle=True), batch_size=7)


    for imgs, targets in train_loader:
        for i, (img, target) in enumerate(zip(imgs,targets)):
            img = reverse_transform(img)
            plt.imshow(img)
            plt.title(str(i)+ ': '+str(target.cpu().numpy()))
            plt.show()



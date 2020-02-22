import numpy as np
from skimage import io
import argparse
import math
from matplotlib import pyplot as plt

class ColorConstancy():
    def __init__(self,verbose=False, thresh_bg=None):
        self.verbose = verbose
        self.thresh_bg = thresh_bg
    def thresh_img(self,img,thresh):
        red_range = thresh[0]!=img[:,:,0]
        green_range = thresh[1]!=img[:,:,1]
        blue_range = thresh[2]!=img[:,:,2]
        valid_range = np.logical_or(red_range, green_range, blue_range)
        return valid_range

    def color_constancy(self,img,preserve_range=True):
        e = np.zeros([3])
        for i in range(3):
            x = img[:,:,i]
            if self.thresh_bg is not None:
                x=x[x!=0]
            e[i]=x.mean()
        if self.verbose: print('channel means',e)
        e=e/math.sqrt(sum(e*e))
        if self.verbose: print('illumination estimate',e)
        d=1/(math.sqrt(3)*e)
        if self.verbose: print('correction coefficient',d)
    #     print(d)
        img_t= img*d
        for i in range(3):
            if self.verbose:
                print('transformed image channel {} max\min: {}\{}'.format(
                i+1,img_t[:,:,i].max(),img_t[:,:,i].min()))
        if preserve_range:
            if self.verbose:
                print('setting values above 255 to 255')
            img_t=img_t.flatten()
            img_t[img_t>255]=255
            img_t=img_t.reshape(img.shape)
        return img_t.astype(np.uint8)


    def compute_cc(self,img,path_skin=None):


        if img.shape[2]>3:
            img=img[:,:,:3]
        if path_skin is not None:
            mask_skin=io.imread(path_skin)
            mask_skin = mask_skin/mask_skin.max()
            if len(mask_skin.shape)<3:
                mask_skin = np.repeat(mask_skin[:, :, np.newaxis], 3, axis=2)
            img = (img*mask_skin).astype(np.uint8)

        if self.thresh_bg is not None:
            mask = self.thresh_img(img,self.thresh)
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            img = img*mask
        img_tx = self.color_constancy(img)
        return img_tx


if __name__=='__main__':
    import os
    from glob import glob
    from tqdm import tqdm
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--src', required=True)
    # parser.add_argument('--dst', required=True)
    # parser.add_argument('--path_skin')
    # color_constancy = ColorConstancy(verbose=True)
    # args = parser.parse_args()
    # color_constancy.compute_cc(args.src, args.dst, args.path_skin)

    # flnames = ['2_viewport1_frame001','4_viewport1_frame001','5_viewport1_frame001',
    #            '6_viewport1_frame001','7_viewport1_frame001','8_viewport1_frame001']
    color_constancy = ColorConstancy(verbose=False,thresh_bg=None)
    # dir_src = r'D:\Data\cGVHD_img_SPIE2018\ConsensusCalls\All_Images'
    # dir_src=r'D:\Data\cGVHD_img_SPIE2018\Longitudinal_Analysis'
    # dir_skin = r'D:\Data\cGVHD_img_SPIE2018\ConsensusCalls\consensus_call_4\extracted_skin'
    # dir_dst = r'D:\Data\cGVHD_img_SPIE2018\ConsensusCalls\All_Images_cc'
    # dir_dst = r'D:\Data\cGVHD_img_SPIE2018\Longitudinal_Analysis'
    dir_src = r'D:\Data\cs-8395-dl\assignment2_data\train'
    dir_dst = r'D:\Data\cs-8395-dl\assignment2_data_cc-gw-not-excld-bg\train'
    # dir_skin = r'D:\Data\cGVHD_img_SPIE2018\ConsensusCalls\Network_Analysis\extracted_skin_edited'
    flpaths=glob(os.path.join(dir_src,'*.jpg'))


    for src in tqdm(flpaths):
        flname = os.path.basename(src).split('.')[0]
        dst = os.path.join(dir_dst, flname+'.jpg')
        # path_skin = os.path.join(dir_skin, flname+'.png')
        img = io.imread(src)
        img_tx=color_constancy.compute_cc(img)
        io.imsave(img_tx,dst)

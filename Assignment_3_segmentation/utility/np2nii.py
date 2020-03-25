import nibabel as nib
import numpy as np
from glob import glob
import os
from tqdm import tqdm
filepaths_np = glob(os.path.join(r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob\*.npy'))
fl_ids = [os.path.basename(path).split('.')[0][-4:] for path in filepaths_np]
dir_src_img = r'D:\Data\cs-8395-dl\assignment3\Testing\img'
dir_src_np= r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob'
dir_dst_nii = r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob'

for fl_id in tqdm(fl_ids):
    path_img = os.path.join(dir_src_img, 'img{}.nii.gz'.format(fl_id))
    path_np = os.path.join(dir_src_np, 'label{}.npy'.format(fl_id))
    path_nii = os.path.join(dir_dst_nii, 'label{}.nii.gz'.format(fl_id))

    img_nib = nib.load(path_img)
    label_np = np.load(path_np)

    label_nib = nib.Nifti1Image(label_np,img_nib.affine, img_nib.header)
    nib.save(label_nib,path_nii)

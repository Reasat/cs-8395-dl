import cc3d
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.pyplot as plt
from plotly import graph_objs as go
import nibabel as nib
import os
from glob import glob
from tqdm import tqdm

filepaths_np = glob(os.path.join(r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob\*.npy'))
file_ids = [os.path.basename(path).split('.')[0][-4:] for path in filepaths_np]

for file_id in tqdm(file_ids):
    path_label= r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob\label{}.nii.gz'.format(file_id)
    label_nib = nib.load(path_label)
    label_3d =label_nib.get_fdata()

    label_bin = label_3d>label_3d.max()/2
    label_bin = label_bin.astype(np.uint16)

    # np.random.seed(42)
    labels_out = cc3d.connected_components(label_bin,
                                           connectivity=6,
                                           out_dtype=np.uint16)

    # You can extract individual components like so:
    N = np.max(labels_out)
    vol_max = 0
    segid_max = 0
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        vol = (extracted_image != 0).sum() / (extracted_image.size)
        # print('segid: {}, Volume ratio: {:.4f}'.format(segid, vol))
        if vol > vol_max:
            segid_max = segid
            print('updating vol_max {:.4f}-->{:.4f} for seg_id {}'.format(vol_max, vol, segid))
            vol_max = vol
    spleen_comp = np.ones_like(labels_out) * (labels_out == segid_max)
    path_save = r'D:\Data\cs-8395-dl\output\assignment3\segmentation_cc_thresh\label{}.nii.gz'.format(file_id)
    label_spleen = nib.Nifti1Image(spleen_comp, label_nib.affine, label_nib.header)
    nib.save(label_spleen, path_save)
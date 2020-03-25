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

filepaths_np = glob(os.path.join(r'D:\Data\cs-8395-dl\output\assignment3\segmentation_prob\*.npy'))
file_ids = [os.path.basename(path).split('.')[0][-4:] for path in filepaths_np]

for file_id in tqdm(file_ids):
    path_axial_prob = r'D:\Data\cs-8395-dl\output\assignment3\axial_spleen_prob\{}.npy'.format(file_id)
    axial_prob = np.load(path_axial_prob)

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
    criteria = 0
    segid_spleen = 0
    for segid in range(1, N + 1):
        extracted_image = labels_out * (labels_out == segid)
        vol = (extracted_image != 0).sum() / (extracted_image.size)
        _, _, z = extract_spleen_coord(extracted_image, label=segid)
        spleen_prob = np.median(axial_prob[z[0]:z[1], 1])
        metric = vol * spleen_prob
        print('segid: {}, Volume ratio: {:.6f}, median prob {:.2f}, vol*prob {:.6f}'.format(segid, vol, spleen_prob,
                                                                                            metric))
        if metric > criteria:
            print('updating criteria {:.6f}-->{:.6f} for seg_id {}'.format(criteria, metric, segid))
            criteria = metric
            segid_spleen = segid

    spleen_comp = np.ones_like(labels_out) * (labels_out == segid_spleen)
    path_save = r'D:\Data\cs-8395-dl\output\assignment3\segmentation_cc_thresh_axial_prob\label{}.nii.gz'.format(file_id)
    label_spleen = nib.Nifti1Image(spleen_comp, label_nib.affine, label_nib.header)
    nib.save(label_spleen, path_save)
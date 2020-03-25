import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import roberts

def recon_img_mask(img_3d_t, mask_3d_t):
    img_3d = img_3d_t[:,0,:,:].detach().cpu().numpy()
    mask_3d = mask_3d_t.squeeze().detach().cpu().numpy()
    return img_3d, mask_3d

def subplot_img_mask(img_3d, mask_3d, ind):
    if hasattr(ind,'__len__'):
        for i in ind:
            print(i)
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            plt.imshow(img_3d[i],cmap='gray')
            plt.subplot(1,2,2)
            mask_slice = mask_3d[i].copy()
            mask_slice[mask_slice!=1]=0
            plt.imshow(mask_slice)
            plt.show()

    else:
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(img_3d[ind], cmap='gray')
        plt.subplot(1, 2, 2)
        mask_slice = mask_3d[ind].copy()
        mask_slice[mask_slice != 1] = 0
        plt.imshow(mask_slice)
        plt.show()
def plot_outline(img_3d, mask_3d, output, ind):
    if hasattr(ind,'__len__'):
        for i in ind:
            plt.figure(figsize=(15, 15))
            plt.imshow(img_3d[i], cmap='gray')
            mask_slice = mask_3d[i].copy()
            mask_slice[mask_slice != 1] = 0
            mask_slice_outline = (roberts(mask_slice > mask_slice.max() / 2) * 255).astype(np.uint8)
            plt.contour(mask_slice_outline, colors='green', linewidths=1)
            if output is not None:
                if np.array_equal(img_3d.shape,output.shape):
                    out_slice = output[i].copy()
                    out_slice_outline = (roberts(out_slice > out_slice.max() / 2) * 255).astype(np.uint8)
                    plt.contour(out_slice_outline, colors='red', linewidths=1)
                else:
                    plt.title('{}'.format(output[i]))

            plt.show()
    else:
        plt.figure(figsize=(15, 15))
        plt.imshow(img_3d[ind], cmap='gray')
        mask_slice = mask_3d[ind].copy()
        mask_slice[mask_slice != 1] = 0
        mask_slice_outline = (roberts(mask_slice > mask_slice.max() / 2) * 255).astype(np.uint8)
        plt.contour(mask_slice_outline, colors='green', linewidths=1)
        if output is not None:
            if np.array_equal(img_3d.shape, output.shape):
                out_slice = output [ind].copy()
                out_slice_outline = (roberts(out_slice > out_slice.max() / 2) * 255).astype(np.uint8)
                plt.contour(out_slice_outline, colors='red', linewidths=1)
            else:
                plt.title('{}'.format(output[ind]))

        plt.show()

def plot_outline_heatmap(img_3d, mask_3d, output, ind, thresh_fac=0.2):
    def plot_slice(img_slice, mask_slice, out_slice=None):
        plt.figure(figsize=(15, 15))
        plt.subplot(1,2,1)
        plt.imshow(img_slice, cmap='gray')
        mask_slice_outline = (roberts(mask_slice > mask_slice.max() / 2) * 255).astype(np.uint8)
        plt.subplot(1,2,2)
        plt.imshow(img_slice, cmap='gray')
        plt.contour(mask_slice_outline, colors='green', linewidths=1)
        if output is not None:
            if np.array_equal(img_3d.shape, output.shape):
                print(out_slice.max(),out_slice.min(), np.median(out_slice))
                plt.imshow(out_slice/out_slice.max(),alpha=0.4, cmap=plt.get_cmap('jet'))
                out_slice_outline = (roberts(out_slice > out_slice.max()*thresh_fac) * 255).astype(np.uint8)
                plt.contour(out_slice_outline, colors='red', linewidths=1)
            else:
                plt.title('{}'.format(output[ind]))

        plt.show()
    for i in ind:
        img_slice = img_3d[i].copy()
        mask_slice = mask_3d[i].copy()
        mask_slice[mask_slice != 1] = 0
        out_slice = None
        if output is not None:
            if np.array_equal(img_3d.shape,output.shape):
                out_slice = output[i].copy()
            else:
                out_slice = np.array(output[i])
        plot_slice(img_slice,mask_slice,out_slice)



def plot_heatmap(img_3d, output_3d, ind):
    if hasattr(ind,'__len__'):
        for i in ind:
            plt.figure(figsize=(15, 15))
            plt.imshow(img_3d[i], cmap='gray')
            out_slice = output_3d[i].copy()
            plt.imshow((out_slice * 255).astype(np.uint8), cmap=plt.get_cmap('jet'), alpha=0.4)
            plt.show()
    else:
        plt.figure(figsize=(15, 15))
        plt.imshow(img_3d[ind], cmap='gray')
        out_slice = output_3d[ind].copy()
        plt.imshow((out_slice * 255).astype(np.uint8), cmap=plt.get_cmap('jet'), alpha=0.4)

        plt.show()
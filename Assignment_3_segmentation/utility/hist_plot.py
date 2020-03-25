import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_hist',default=r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\history')
parser.add_argument('--filename', required= True)
parser.add_argument('--include',nargs='+')
parser.add_argument('--epoch_range',nargs='+',type =int)
# dir_hist='D:\\Data\\cGVHD_img_SPIE2018\\logs'
args = parser.parse_args()
FONTSIZE=15
filepath_log=os.path.join(args.dir_hist,args.filename)
log=pickle.load(open(filepath_log,'rb'))
for key in log.keys():
    if hasattr(log[key],'__len__'):
        if len(log[key])==0 :
            print('key "{}" empty'.format(key))
        else:
            if args.include is not None:
                for key_inc in args.include:
                    if key_inc in key:
                        if args.epoch_range is not None:
                            plt.plot(range(args.epoch_range[1]-args.epoch_range[0]),log[key][args.epoch_range[0]:args.epoch_range[1]], label = key)
                            print('{} : maximum {:.4f} at epoch {}, minimum {:.4f} at epoch {}, latest value {:.4f} '.format(
                                key,
                                np.max(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                                np.argmax(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                                np.min(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                                np.argmin(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                                log[key][args.epoch_range[0]:args.epoch_range[1]][-1]))
                        else:
                            plt.plot(range(len(log[key])),log[key], label = key)
                            plt.xlabel('Epoch')
                            print('{} : maximum {:.4f} at epoch {}, minimum {:.4f} at epoch {}, latest value {:.4f} '.format(
                                key,np.max(log[key]),
                                np.argmax(log[key]),
                                np.min(log[key]),
                                np.argmin(log[key]),
                                log[key][-1])
                            )
            else:
                if args.epoch_range is not None:
                    plt.plot(range(args.epoch_range[1]-args.epoch_range[0]),log[key][args.epoch_range[0]:args.epoch_range[1]], label = key)
                    print('{} : maximum {:.4f} at epoch {}, minimum {:.4f} at epoch {}, latest value {:.4f} '.format(
                        key,
                        np.max(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                        np.argmax(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                        np.min(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                        np.argmin(log[key][args.epoch_range[0]:args.epoch_range[1]]),
                        log[key][args.epoch_range[0]:args.epoch_range[1]][-1]))
                else:
                    plt.plot(range(len(log[key])),log[key], label = key)
                    print('{} : maximum {:.4f} at epoch {}, minimum {:.4f} at epoch {}, latest value {:.4f} '.format(
                        key,
                        np.max(log[key]),
                        np.argmax(log[key]),
                        np.min(log[key]),
                        np.argmin(log[key]),
                        log[key][-1]))

plt.legend()
plt.show()

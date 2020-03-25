from glob import glob
import os
from albumentations import (
    Compose, Resize, Normalize, RandomBrightnessContrast,
    HorizontalFlip,RandomRotate90,RandomCrop,
CenterCrop
)
import albumentations.pytorch as albu_torch
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import Spleen_Dataset
from logger import Logger
from loss import FTL, bceWithSoftmax
from torch.utils.data import DataLoader
from models import DPN92, ResNet18
import torch.optim as optim
import torch
from torch import nn
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from metrics import get_acc,get_recall
from tqdm import tqdm
from plotters import recon_img_mask, plot_outline, plot_heatmap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STAMP=time.strftime('%Y-%m-%d-%H-%M-%S')
from metrics import dice
from average_tracker import AverageMeter
from sklearn.metrics import roc_auc_score


parser=argparse.ArgumentParser()
parser.add_argument('--dir_project', help='project directory', default=r'..')
parser.add_argument('--dir_lf', help='directory large fileIDs',default=r'D:\Data\cs-8395-dl')
parser.add_argument('--folderData', help='data directory', default=r'assignment3\Training')
parser.add_argument('--folderPartition', help='partition directory', default='train_test_org')
parser.add_argument('--path_kfold', help='kfold', default=r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\kfold_5.bin')
parser.add_argument('--encoder',help='encoder')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--batchSize', help='batch size', type=int, default=32)
parser.add_argument('--epoch', help='epoch', type=int, default=24)
parser.add_argument('--lossWeight', help='ce', type=float, nargs='+', default= [0.3, 0.7])
parser.add_argument('--resume_from', help='filepath to resume training')
parser.add_argument('--bottleneckFeatures', help='bottleneck the encoder Features', type=int, default=1)
parser.add_argument('--overrideLR', help='override LR from resumed network', type=int, default=1)
parser.add_argument('--brightness',nargs='+', type=float)
parser.add_argument('--contrast',nargs='+', type=float)
parser.add_argument('--cropSize', type=int)
parser.add_argument('--resize', type=int)
parser.add_argument('--to_ram',type=int, default=0)
parser.add_argument('--loss_weights',nargs='+', type=float)
parser.add_argument('--msg')
parser.add_argument('--resume_training')
parser.add_argument('--extract_spleen', default=False)
args=parser.parse_args()

# setting up directories
DIR_LF = args.dir_lf#r'D:\Data\cs-8395-dl'
dir_data = os.path.join(DIR_LF,args.folderData) #os.path.join(DIR_LF,'assignment1_data')
dir_model = os.path.join(args.dir_lf, 'model',TIME_STAMP)
dir_history = os.path.join(args.dir_project, 'history')
dir_log = os.path.join(args.dir_project, 'log')
dir_config = os.path.join(args.dir_project, 'config')

if os.path.exists(dir_history) is False:
    os.mkdir(dir_history)
if os.path.exists(dir_log) is False:
    os.mkdir(dir_log)
if os.path.exists(dir_config) is False:
    os.mkdir(dir_config)
if os.path.exists(os.path.join(args.dir_lf, 'model')) is False:
    os.mkdir(os.path.join(args.dir_lf, 'model'))


filepath_log = os.path.join(dir_log, '{}.log'.format(TIME_STAMP))
filepath_cfg = os.path.join(dir_config, '{}.cfg'.format(TIME_STAMP))

sys.stdout = Logger(filepath_log)
print(TIME_STAMP)
print(os.path.basename(__file__))
config=vars(args)
config_ls=sorted(list(config.items()))
print('--------------------------------------------------------------------------------------------------------------------')
for item in config_ls:
    print('{}: {}'.format(item[0],item[1]))
print('--------------------------------------------------------------------------------------------------------------------')
with open(filepath_cfg, 'w') as file:
    for item in config_ls:
        file.write('{}: {}\n'.format(item[0], item[1]))

if os.path.exists(dir_model)==0:
    print('creating directory to save model at {}'.format(dir_model))
    os.mkdir(dir_model)


dir_data_train = os.path.join(dir_data, 'Training')
with open(os.path.join(args.dir_project,'partition',args.folderPartition,'Training.txt')) as  f:
    file_ids = [id.strip() for id in f.readlines()]
# get train filenames
kf = pickle.load(open(args.path_kfold,'rb'))
BATCH_SIZE=args.batchSize
LR = args.lr
EPOCH=args.epoch

compute_loss = bceWithSoftmax(weights=args.lossWeight)

for i_fold, (ind_train, ind_valid) in  enumerate(kf.split(file_ids)):
    # ind_train= [ind_train[0]] # debug
    # ind_valid = [ind_valid[0]]
    filepath_hist = os.path.join(dir_history, '{}_fold-{}.bin'.format(TIME_STAMP,i_fold+1))
    print('executing fold {}/{}'.format(i_fold+1,kf.n_splits))

    file_ids_train = np.array(file_ids)[ind_train]
    file_ids_valid = np.array(file_ids)[ind_valid]
    print('training patients ', file_ids_train)
    print('validation patients ', file_ids_valid)
    # file_ids_train = np.array(['0001'])
    # model = DPN92().to(device)
    model = ResNet18(pretrained=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    model_save_criteria = np.inf

    filepath_model_best = os.path.join(dir_model, '{}_{}_fold-{}_best.pt'.format(TIME_STAMP, i_fold+1, args.encoder))  ##
    filepath_model_latest = os.path.join(dir_model, '{}_{}_fold-{}_latest.pt'.format(TIME_STAMP, i_fold+1, args.encoder))  ##

    if args.resume_training is not None:
        train_states = torch.load(args.resume_training)
        model.load_state_dict(train_states['model_state_dict'])
        optimizer.load_state_dict(train_states['optimizer_state_dict'])
        print('loaded model weights from {}\nepoch {}'.format(args.resume_training, train_states['epoch']))
        print('loss of pretrained model: {:.4f}'.format(model_save_criteria))

    Dataset_train = Spleen_Dataset(
        dir_data=dir_data,
        fileIDs=file_ids_train,
        # extract_spleen=args.extract_spleen
        # to_ram=True
    )
    Dataset_test = Spleen_Dataset(
        dir_data=dir_data,
        fileIDs=file_ids_valid,
        # extract_spleen=args.extract_spleen
        # to_ram=True
    )
    idx_train = np.arange(len(Dataset_train))
    idx_valid = np.arange(len(Dataset_test))

    loss_ce_train = []
    loss_ce_valid = []
    metric_acc_train = []
    metric_acc_valid = []

    loss_ce_train_am = AverageMeter(name='loss_ce')
    metric_acc_train_am = AverageMeter(name='acc')

    loss_ce_valid_am = AverageMeter(name='loss_ce')
    metric_acc_valid_am = AverageMeter(name='metric_auc')

    for epoch in tqdm(range(args.epoch)):
        np.random.shuffle(idx_train)
        loss_ce_train_am.reset()
        metric_acc_train_am.reset()
        metric_acc_valid_am.reset()
        loss_ce_valid_am.reset()
        for i_p, ind in enumerate(idx_train):
            # print('train samples {}'.format(len(Dataset)))
            _ = model.train()  # Set model to training mode

            sample = Dataset_train[ind] # get patient
            batch_start_range = range(0, sample[0].shape[-3], BATCH_SIZE)
            batch_start_range = np.array(list(batch_start_range))
            np.random.shuffle(batch_start_range) # randomize batch start index

            # print('accumulating gradients, acc_step = {}'.format(acc_step))
            for i_b, batch_start in tqdm(enumerate(batch_start_range)):
                # sample[0] dim [147, 512, 512], sample[1] [147, 512, 512]
                image_batch = sample[0][batch_start:batch_start + BATCH_SIZE,:, :]
                image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)

                image_batch = image_batch.float().to(device)
                mask_batch = sample[1][batch_start:batch_start + BATCH_SIZE, :, :]
                target_batch = ((torch.sum(mask_batch, dim=(1, 2)) > 0) * 1)
                target_batch = target_batch.to(device)
                output = model(image_batch)


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


                loss = compute_loss(output, target_batch)
                acc=get_acc(target_batch.cpu(),output.cpu())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_ce_train_am.update(loss.item())
                metric_acc_train_am.update(acc)

            print(
                'train >>> fold {}/{}, epoch: {}/{}, patient {}/{}, mean_loss {:.4f}, mean_acc {:.4f}'.format(
                    i_fold + 1,
                    kf.n_splits,
                    epoch + 1,
                    args.epoch,
                    i_p + 1,
                    len(file_ids_train),
                    loss_ce_train_am.avg,
                    metric_acc_train_am.avg
                ))
        # auc = roc_auc_score(target_all, output_all)

        with torch.no_grad():
            for ind in tqdm(idx_valid):
                sample=Dataset_test[ind]
                output_3d = []
                mask_3d = []
                batch_start_range = range(0, sample[0].shape[-3], BATCH_SIZE)
                batch_start_range = np.array(list(batch_start_range))
                # batch_start_range = batch_start_range[batch_start_range > 120]
                # batch_start_range = batch_start_range[batch_start_range < 130]
                # print(batch_start_range)
                for i_b, batch_start in tqdm(enumerate(batch_start_range)):
                    # sample[0] dim [147, 512, 512]
                    image_batch = sample[0].squeeze()[batch_start:batch_start + BATCH_SIZE,:, :]
                    image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)
                    image_batch = image_batch.float().to(device)

                    mask_batch = sample[1][batch_start:batch_start + BATCH_SIZE, :, :]
                    target_batch = ((torch.sum(mask_batch, dim=(1, 2)) > 0) * 1)
                    target_batch = target_batch.to(device)
                    output = model(image_batch)

                    # plot
                    # if epoch == args.epoch // 2 and i_p == 0 and i_b == 0:
                    #     image_3d_rec, mask_3d_rec = recon_img_mask(image_batch, mask_batch)
                    #     _, output_3d_rec = recon_img_mask(image_batch, output)
                    #     plot_heatmap(image_3d_rec, output_3d_rec,
                    #                  range(0, output_3d_rec.shape[0], 2))
                    #     output_3d_rec_bin = (output_3d_rec > output_3d_rec.max() / 2) * 1
                    #     print('out_max', output_3d_rec_bin.max())
                    #     plot_outline(image_3d_rec, mask_3d_rec, output_3d_rec_bin,
                    #                  range(0, output_3d_rec_bin.shape[0], 2))

                    loss = compute_loss(output, target_batch)
                    acc = get_acc(target_batch.cpu(), output.cpu())

                    # print(output_3d.shape)

                    loss_ce_valid_am.update(loss.item())
                    metric_acc_valid_am.update(acc)
        print(
            'valid >>> epoch: {}/{}, mean_loss: {:.4f}, mean_acc: {:.4f}'.format(
                epoch + 1,
                args.epoch,
                loss_ce_valid_am.avg,
                metric_acc_valid_am.avg
            ))
        loss_ce_train = []
        loss_ce_valid = []
        metric_acc_train = []
        metric_acc_valid = []

        loss_ce_train.append(loss_ce_train_am.avg)
        metric_acc_train.append(metric_acc_train_am.avg)
        loss_ce_valid.append(loss_ce_valid_am.avg)
        metric_acc_valid.append(metric_acc_valid_am.avg)

        # save train history
        log = {
            'loss_train':loss_ce_train,
            'metric_acc_train': metric_acc_train,
            'loss_ce_train': loss_ce_train,
            'loss_valid': loss_ce_valid,
            'metric_acc_valid': metric_acc_valid,

        }
        with open(filepath_hist, 'wb') as pfile:
            pickle.dump(log, pfile)

        # save best model
        if np.mean(loss_ce_valid_am.avg)<model_save_criteria:

            print('criteria decreased from {:.4f} to {:.4f}, saving best model at {}'.format(model_save_criteria,
                                                                                             np.mean(loss_ce_valid_am.avg),
                                                                                             filepath_model_best))

            model_save_criteria=loss_ce_valid_am.avg
            train_states = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_save_criteria': model_save_criteria,
            }
            torch.save(train_states, filepath_model_best)
    # save latest model
    train_states = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_save_criteria':loss_ce_valid_am.avg,
    }
    torch.save(train_states, filepath_model_latest)
    # #
print(TIME_STAMP)





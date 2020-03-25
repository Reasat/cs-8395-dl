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
from dataloader import Spleen_Dataset, randomize_batch
from logger import Logger
from loss import FTL, dice_loss
from torch.utils.data import DataLoader
from models import UNet11, AlbuNet
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

parser=argparse.ArgumentParser()
parser.add_argument('--dir_project', help='project directory', default=r'..')
parser.add_argument('--dir_lf', help='directory large fileIDs',default=r'D:\Data\cs-8395-dl')
parser.add_argument('--folderData', help='data directory', default=r'assignment3\Training')
parser.add_argument('--folderPartition', help='partition directory', default='train_test_org')
parser.add_argument('--path_kfold', help='kfold', default=r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\kfold_5.bin')
parser.add_argument('--encoder',help='encoder',default='resnet34')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--batchSize', help='batch size', type=int, default=4)
parser.add_argument('--epoch', help='epoch', type=int, default=40)
parser.add_argument('--lossWeight', help='-dice+ce', type=float, nargs='+', default= [0.2, 0.8])
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
parser.add_argument('--extract_spleen', default=False)
parser.add_argument('--transforms', nargs='+')
parser.add_argument('--axis')
args=parser.parse_args()

def model_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
    return model
def model_train(model):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = True
    return model
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

criterion_ce = nn.BCEWithLogitsLoss()

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
    model = AlbuNet(pretrained=True, is_deconv=True).to(device)
    # model = UNet11(pretrained=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    model_save_criteria = np.inf

    filepath_model_best = os.path.join(dir_model, '{}_{}_fold-{}_best.pt'.format(TIME_STAMP,  args.encoder,i_fold+1))  ##
    filepath_model_latest = os.path.join(dir_model, '{}_{}_fold-{}_latest.pt'.format(TIME_STAMP, args.encoder,i_fold+1 ))  ##

    if args.resume_from is not None:
        train_states = torch.load(args.resume_from)
        model.load_state_dict(train_states['model_state_dict'])
        optimizer.load_state_dict(train_states['optimizer_state_dict'])
        print('loaded model weights from {}\nepoch {}'.format(args.resume_from, train_states['epoch']))
        print('loss of pretrained model: {:.4f}'.format(train_states['model_save_criteria']))
        del train_states

    Dataset_train = Spleen_Dataset(
        dir_data=dir_data,
        fileIDs=file_ids_train,
        extract_spleen=args.extract_spleen,
        transform=args.transforms,
        axis = args.axis
        # to_ram=True
    )
    Dataset_test = Spleen_Dataset(
        dir_data=dir_data,
        fileIDs=file_ids_valid,
        extract_spleen=args.extract_spleen,
        axis=args.axis
        # to_ram=True
    )
    idx_train = np.arange(len(Dataset_train))
    idx_valid = np.arange(len(Dataset_test))

    loss_train = []
    loss_train_am = AverageMeter(name='loss')
    loss_d_train = []
    loss_d_train_am = AverageMeter(name='loss_dice')
    loss_ce_train = []
    loss_ce_train_am = AverageMeter(name='loss_ce')



    loss_valid = []
    loss_d_valid = []
    loss_ce_valid = []

    metric_d_valid = []

    for epoch in tqdm(range(args.epoch)):
        np.random.shuffle(idx_train) # randomize patient
        loss_train_am.reset()
        loss_d_train_am.reset()
        loss_ce_train_am.reset()

        for i_p, ind in enumerate(idx_train):
            # print('train samples {}'.format(len(Dataset)))
            model = model_train(model)  # Set model to training mode

            sample = Dataset_train[ind]
            # randomize axial position of samples
            sample = randomize_batch(sample[0], sample[1])

            batch_start_range = range(0, sample[0].shape[-3], BATCH_SIZE)
            batch_start_range = np.array(list(batch_start_range))

            for i_b, batch_start in tqdm(enumerate(batch_start_range)):
                # sample[0] dim [1, 147, 512, 512]
                image_batch = sample[0].squeeze()[batch_start:batch_start + BATCH_SIZE,:, :]
                image_batch = torch.cat(3 * [image_batch.unsqueeze(1)], dim=1)

                image_batch = image_batch.float().to(device)

                mask_batch = sample[1].squeeze()[batch_start:batch_start + BATCH_SIZE,:, :]
                mask_batch = mask_batch.unsqueeze(1)
                mask_batch = mask_batch.float().to(device)

                output = model(image_batch)

                # plot
                # if epoch+1 == args.epoch and i_p == 0 and i_b == 0:
                #     image_3d_rec, mask_3d_rec = recon_img_mask(image_batch, mask_batch)
                #     _, output_3d_rec = recon_img_mask(image_batch, output)
                #     plot_heatmap(image_3d_rec, output_3d_rec,
                #                  range(0, output_3d_rec.shape[0], 2))
                #     output_3d_rec_bin = (output_3d_rec > output_3d_rec.max() / 2) * 1
                #     print('out_max',output_3d_rec_bin.max())
                #     plot_outline(image_3d_rec, mask_3d_rec, output_3d_rec_bin,
                #                  range(0, output_3d_rec_bin.shape[0], 2))


                loss_d = dice_loss(torch.sigmoid(output), mask_batch)
                loss_ce = criterion_ce(output, mask_batch)
                loss = args.lossWeight[0] * loss_d + args.lossWeight[1] * loss_ce
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                loss_train_am.update(loss.item())
                loss_d_train_am.update(loss_d.item())
                loss_ce_train_am.update(loss_ce.item())

            print(
                'train >>> fold {}/{}, epoch: {}/{}, patient {}/{}, mean_loss {:.4f}, mean_dice_loss: {:.4f}, mean_ce: {:.4f}'.format(
                    i_fold+1,
                    kf.n_splits,
                    epoch + 1,
                    args.epoch,
                    i_p + 1,
                    len(file_ids_train),
                    loss_train_am.avg,
                    loss_d_train_am.avg,
                    loss_ce_train_am.avg
                ))


        dice_coef = []
        loss_valid_all=[]
        loss_d_valid_all=[]
        loss_ce_valid_all=[]
        model = model_eval(model)
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

                    mask_batch = sample[1].squeeze()[batch_start:batch_start + BATCH_SIZE,:, :]
                    mask_batch = mask_batch.unsqueeze(1)
                    mask_batch = mask_batch.float().to(device)

                    output_batch = model(image_batch)

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

                    output_3d.append((output_batch).squeeze(1).detach().cpu().numpy())
                    mask_3d.append(mask_batch.squeeze(1).detach().cpu().numpy())
                # print(len(output_3d), output_3d[0].shape, output_3d[11].shape)
                output_3d = np.concatenate(np.array(output_3d), axis=0)
                mask_3d = np.concatenate(np.array(mask_3d), axis=0)
                output_3d_sig = torch.sigmoid(torch.tensor(output_3d)).numpy()
                # print(torch.tensor(output_3d).shape, torch.tensor(mask_3d).shape)
                loss_d = dice_loss(torch.tensor(output_3d_sig), torch.tensor(mask_3d))
                loss_ce = criterion_ce(torch.tensor(output_3d), torch.tensor(mask_3d))
                loss = args.lossWeight[0] * loss_d + args.lossWeight[1] * loss_ce

                # print(output_3d.shape)

                loss_valid_all.append(loss.item())
                loss_d_valid_all.append(loss_d.item())
                loss_ce_valid_all.append(loss_ce.item())
                dice_coef.append(dice(
                    ( output_3d_sig> output_3d_sig.max()/2) * 1
                    , mask_3d
                ))

                del output_3d,output_3d_sig, mask_3d,
        print(
            'valid >>> epoch: {}/{}, mean_loss: {:.4f}, mean_dice: {:.4f}'.format(
                epoch + 1,
                args.epoch,
                np.mean(loss_valid_all),
                np.mean(dice_coef)
            ))


        loss_train.append(loss_train_am.avg)
        loss_d_train.append(loss_d_train_am.avg)
        loss_ce_train.append(loss_ce_train_am.avg)
        loss_valid.append(np.mean(loss_valid_all))
        loss_d_valid.append(np.mean(loss_d_valid_all))
        loss_ce_valid.append(np.mean(loss_ce_valid_all))
        metric_d_valid.append(np.mean(dice_coef))

        # save train history
        log = {
            'loss_train':loss_train,
            'loss_dice_train': loss_d_train,
            'loss_ce_train': loss_ce_train,
            'loss_valid': loss_valid,
            'loss_ce_valid': loss_ce_valid,
            'loss_dice_valid': loss_d_valid,
            'metric_dice_valid': metric_d_valid,

        }
        with open(filepath_hist, 'wb') as pfile:
            pickle.dump(log, pfile)

        # save best model
        if np.mean(loss_valid_all)<model_save_criteria:

            print('criteria decreased from {:.4f} to {:.4f}, saving best model at {}'.format(model_save_criteria,
                                                                                             np.mean(loss_valid_all),
                                                                                             filepath_model_best))

            model_save_criteria=np.mean(loss_valid_all)
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
            'model_save_criteria': np.mean(loss_valid_all),
        }
        torch.save(train_states, filepath_model_latest)
        del train_states
    print('end of fold, deleting model')
    del model
print(TIME_STAMP)





from glob import glob
import os
from albumentations import (
    Compose, Resize, Normalize, RandomBrightnessContrast, HorizontalFlip,
CenterCrop
)
import albumentations.pytorch as albu_torch
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import ISIC_Dataset
from logger import Logger
from loss import bceWithSoftmax
from torch.utils.data import DataLoader
from models import ResNet18, ResNet50, DPN92
import torch.optim as optim
import torch
import time
import argparse
import numpy as np
import pickle
import pandas as pd
from metrics import get_acc,get_recall

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STAMP=time.strftime('%Y-%m-%d-%H-%M-%S')

parser=argparse.ArgumentParser()
parser.add_argument('--dir_project', help='project directory', default=r'..')
parser.add_argument('--dir_lf', help='directory large files',default=r'D:\Data\cs-8395-dl')
parser.add_argument('--folderData', help='data directory', default='assignment2_data')
parser.add_argument('--encoder',help='encoder',default='resnet18')
parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
parser.add_argument('--batchSize', help='batch size', type=int, default=32)
parser.add_argument('--epoch', help='epoch', type=int, default=400)
parser.add_argument('--resume_from', help='filepath to resume training')
parser.add_argument('--bottleneckFeatures', help='bottleneck the encoder Features', type=int, default=1)
parser.add_argument('--overrideLR', help='override LR from resumed network', type=int, default=1)
parser.add_argument('--brightness',nargs='+', type=float)
parser.add_argument('--contrast',nargs='+', type=float)
parser.add_argument('--resize', type=int)
parser.add_argument('--to_ram',type=int, default=0)
parser.add_argument('--loss_weights',nargs='+')
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

filepath_hist = os.path.join(dir_history, '{}.bin'.format(TIME_STAMP))
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

filepath_model_best = os.path.join(dir_model, '{}_{}_best.pt'.format(TIME_STAMP, args.encoder))  ##
filepath_model_latest = os.path.join(dir_model, '{}_{}_latest.pt'.format(TIME_STAMP, args.encoder))  ##

dir_data_train = os.path.join(dir_data, 'train')
dir_data_test = os.path.join(dir_data, 'test')

# get train filenames
filepath_train_label = os.path.join(dir_data, 'labels','Train_labels.csv')
df_train = pd.read_csv(filepath_train_label)
df_train.set_index('image',inplace=True)
files_train = df_train.index.values
labels_train_one_hot=[df_train.loc[flname].values for flname in files_train]
labels_train_cat = [np.argmax(label) for label in labels_train_one_hot]
# get test filenames
filepath_test_label = os.path.join(dir_data, 'labels','Test_labels.csv')
df_test = pd.read_csv(filepath_test_label)
df_test.set_index('image',inplace=True)
files_test = df_test.index.values
labels_test_one_hot=[df_test.loc[flname].values for flname in files_test]
labels_test_cat = [np.argmax(label) for label in labels_test_one_hot]

# Dataloader Parameters
aug ={
    'train': Compose([
    HorizontalFlip(),
    RandomBrightnessContrast(
        brightness_limit=args.brightness,
        contrast_limit=args.contrast,
    ),
    CenterCrop(args.resize, args.resize, p=0.5),
    Resize(args.resize,args.resize),
    Normalize(),
    albu_torch.ToTensorV2()
    ]),
    'valid': Compose([
    Resize(args.resize,args.resize),
    Normalize(),
    albu_torch.ToTensorV2()
    ])
}

BATCH_SIZE=args.batchSize
LR = args.lr
EPOCH=args.epoch
Dataset_train = ISIC_Dataset(dir_data=dir_data_train, files=df_train.index.values, label_cat=labels_train_cat, transform=aug['train'])
loader_train=DataLoader(Dataset_train,batch_size=BATCH_SIZE, shuffle=True)
print('train samples {}'.format(len(Dataset_train)))
Dataset_valid = ISIC_Dataset(dir_data=dir_data_test, files=df_test.index.values, label_cat=labels_test_cat, transform=aug['valid'])
loader_valid=DataLoader(Dataset_valid,batch_size=BATCH_SIZE, shuffle=False)
print('validation samples {}'.format(len(Dataset_valid)))
# Model
if args.encoder == 'resnet18':
    model = ResNet18(pretrained=True, bottleneckFeatures=args.bottleneckFeatures).to(device)
if args.encoder == 'resnet50':
    model = ResNet50(pretrained=True, bottleneckFeatures=args.bottleneckFeatures).to(device)
if args.encoder == 'dpn92':
    model = DPN92().to(device)

# print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)

# Train
if args.resume_from is not None:
    # Resume?
    print('resuming training from {}'.format(args.resume_from))
    train_states = torch.load(args.resume_from)
    model.load_state_dict(train_states['model_state_dict'])
    if args.overrideLR==0:
        optimizer.load_state_dict(train_states['optimizer_state_dict'])
    epoch_range = np.arange(train_states['epoch']+1, train_states['epoch']+1+EPOCH)
else:
    train_states = {
                'epoch': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_save_criteria': np.inf,
            }
    epoch_range = np.arange(1,EPOCH+1)

loss_train=[]
loss_valid=[]
acc_train = []
acc_valid=[]
recall_macro_valid = []
recall_micro_valid = []
compute_loss = bceWithSoftmax(weights=args.loss_weights)
for epoch in epoch_range:
    running_loss = 0
    running_acc = 0
    model.train()
    for i, sample in enumerate(loader_train):
        optimizer.zero_grad()
        img = sample[0].to(device)
        target = sample[1].to(device)
        output = model(img)
        # print(target,output)
        loss = compute_loss(output,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += get_acc(target.cpu(),output.cpu())
        mean_loss = running_loss / (i + 1)
        mean_acc = running_acc / (i + 1)
        print('train >>> epoch: {}/{}, batch: {}/{}, mean_loss: {:.4f}, mean_acc: {:.4f}'.format(
            epoch,
            epoch_range[-1],
            i+1,
            len(loader_train),
            mean_loss,
            mean_acc

        ))
    loss_train.append(mean_loss)
    acc_train.append(mean_acc)
    model.eval()
    running_loss = 0
    output_all=torch.FloatTensor([])
    target_all=torch.FloatTensor([])
    with torch.no_grad():
        for i, sample in enumerate(loader_valid):
            img = sample[0].to(device)
            target = sample[1].to(device)
            output = model(img)
            output_all=torch.cat((output_all,output.float().cpu()),dim=0)
            target_all=torch.cat((target_all,target.float().cpu()),dim=0)
            loss = compute_loss(output,target)
            running_loss += loss.item()
            running_acc += get_acc(target.cpu(),output.cpu())
            mean_loss = running_loss / (i + 1)

    recall_macro = get_recall(target_all, output_all, average='macro')
    recall_micro = get_recall(target_all, output_all, average='micro')
    mean_acc=get_acc(target_all, output_all)
    acc_valid.append(mean_acc)
    print('valid >>> epoch: {}/{}, mean_loss: {:.4f}, mean_acc: {:.4f}'.format(
        epoch,
        epoch_range[-1],
        mean_loss,
        mean_acc
    ))
    print('recall_micro_valid: {:.4f}, recall_macro_valid: {:4f}'.format(recall_micro, recall_macro))

    loss_valid.append(mean_loss)
    recall_macro_valid.append(recall_macro)
    recall_micro_valid.append(recall_micro)
    # save train history
    log = {
        'loss_train':loss_train,
        'loss_valid':loss_valid,
        'acc_train': acc_train,
        'acc_valid': acc_valid,
        'recall_micro_valid':recall_micro_valid,
        'recall_macro_valid':recall_macro_valid
        }
    with open(filepath_hist, 'wb') as pfile:
        pickle.dump(log, pfile)

    # save best model
    if mean_loss<train_states['model_save_criteria']:
        print('criteria decreased from {:.4f} to {:.4f}, saving best model at {}'.format(train_states['model_save_criteria'],
                                                                                         mean_loss,
                                                                                         filepath_model_best))
        train_states = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_save_criteria': mean_loss,
        }
        torch.save(train_states, filepath_model_best)

# save latest model
train_states = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_save_criteria': mean_loss,
}
torch.save(train_states, filepath_model_latest)

print(TIME_STAMP)





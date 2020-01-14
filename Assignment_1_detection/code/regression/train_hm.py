from glob import glob
import os
from albumentations import Compose, Normalize, Resize
import albumentations.pytorch as albu_torch
import sys
sys.path.insert(1,r'..\utility')
sys.path.insert(1,r'..\models')
from dataloader import  Mobile_Dataset_HM_RAM,reverse_transform
from logger import Logger
from loss import fl
from torch.utils.data import DataLoader
from models import HM_Net, UNet11
import torch.optim as optim
import torch
import time
import argparse
import numpy as np
import pickle
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIME_STAMP=time.strftime('%Y-%m-%d-%H-%M-%S')

parser=argparse.ArgumentParser()
parser.add_argument('--dir_project', help='project directory', required=True)
parser.add_argument('--dir_lf', help='directory large files', required=True)
parser.add_argument('--folderData', help='data directory', required=True)
parser.add_argument('--encoder',help='encoder',required=True)
parser.add_argument('--lr', help='learning rate', type=float, required=True)
parser.add_argument('--batchSize', help='batch size', type=int, required=True)
parser.add_argument('--epoch', help='epoch', type=int, required= True)
parser.add_argument('--resume_from', help='filepath to resume training')
parser.add_argument('--bottleneckFeatures', help='bottleneck the encoder Features', type=int, default=1)
parser.add_argument('--alpha', help='loss parameter', type=float, required=True)
parser.add_argument('--gamma', help='loss parameter', type=int, required=True)

args=parser.parse_args()

# setting up directories
DIR_LF = args.dir_lf#r'D:\Data\cs-8395-dl'
dir_data = os.path.join(DIR_LF,args.folderData) #os.path.join(DIR_LF,'assignment1_data')
dir_model = os.path.join(args.dir_lf, 'model',TIME_STAMP)
dir_history = os.path.join(args.dir_project, 'history')
dir_log = os.path.join(args.dir_project, 'log')


filepath_hist = os.path.join(dir_history, '{}.bin'.format(TIME_STAMP))
filepath_log = os.path.join(dir_log, '{}.log'.format(TIME_STAMP))
filepath_cfg = os.path.join(args.dir_project, 'config', '{}.cfg'.format(TIME_STAMP))

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

dir_data_train = os.path.join(dir_data, 'train')
filepaths_train = glob(os.path.join(dir_data_train, '*.jpg'))
flnames_train = [os.path.basename(path)for path in filepaths_train]

dir_data_valid = os.path.join(dir_data, 'validation')
filepaths_valid = glob(os.path.join(dir_data_valid, '*.jpg'))
flnames_valid = [os.path.basename(path)for path in filepaths_valid]

filepath_labels = os.path.join(dir_data, 'labels', 'labels.txt')
with open(filepath_labels, 'r') as f:
    label_data = f.readlines()
label_dict = {}
for data in label_data:
    name, x, y = data.strip().split(' ')
    label_dict[name] = (float(x), float(y))

# Dataloader
aug = Compose([
    Resize(256,256),
    #            RandomRotate90(),
    Normalize(),
    albu_torch.ToTensorV2()
],
)
BATCH_SIZE=args.batchSize
LR = args.lr
EPOCH=args.epoch
Dataset_train = Mobile_Dataset_HM_RAM(dir_data=dir_data_train,files=flnames_train,label_dict=label_dict,transform=aug)
loader_train=DataLoader(Dataset_train,batch_size=BATCH_SIZE, shuffle=True)
print('train samples {}'.format(len(Dataset_train)))
Dataset_valid = Mobile_Dataset_HM_RAM(dir_data=dir_data_valid,files=flnames_valid,label_dict=label_dict,transform=aug)
loader_valid=DataLoader(Dataset_valid,batch_size=BATCH_SIZE, shuffle=True)
print('validation samples {}'.format(len(Dataset_valid)))
# Model
# if args.encoder == 'resnet18':
#     model = ResNet18(pretrained=True, bottleneckFeatures=args.bottleneckFeatures).to(device)
if args.encoder =='hm_net':
    model = HM_Net().to(device)
if args.encoder =='unet-vgg11':
    model = UNet11(pretrained=True).to(device)
print(model)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)

# Train
if args.resume_from is not None:
    # Resume?
    print('resuming training from {}'.format(args.resume_from))
    train_states = torch.load(args.resume_from)
    model.load_state_dict(train_states['model_state_dict'])
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
for epoch in epoch_range:
    running_loss = 0
    model.train()
    for i, sample in enumerate(loader_train):
        optimizer.zero_grad()
        img = sample[0].to(device)
        target = sample[1].to(device)
        output = torch.sigmoid(model(img))
        # print(target.max(),output.max())
        loss = fl(target,output,alpha=args.alpha,gamma=args.gamma)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        mean_loss = running_loss / (i + 1)
        print('train >>> epoch: {}/{}, batch: {}/{}, mean_loss: {:.4f}'.format(
            epoch,
            epoch_range[-1],
            i+1,
            len(loader_train),
            mean_loss

        ))

        if epoch%20==0:
            for i in range(len(img)):
                img_r = reverse_transform(img[i].cpu().squeeze())
                hm = output[i].detach().cpu().numpy().squeeze()
                trgt = target[i].detach().cpu().numpy().squeeze()
                # print(img_r.shape)
                plt.subplot(1, 3, 1)
                plt.imshow(img_r)
                # plt.plot(target.cpu().squeeze())
                plt.plot(hm, alpha=0.3)
                plt.subplot(1, 3, 2)
                plt.imshow(trgt)
                plt.subplot(1, 3, 3)
                plt.imshow(hm)
                plt.show()

    loss_train.append(mean_loss)
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, sample in enumerate(loader_valid):
            img = sample[0].to(device)
            target = sample[1].to(device)
            output = torch.sigmoid(model(img))
            loss = fl(target,output,alpha=args.alpha,gamma=args.gamma)
            # print(loss.item())
            running_loss += loss.item()
            mean_loss = running_loss / (i + 1)
            if epoch % 20 == 0:
                for i in range(len(img)):
                    img_r = reverse_transform(img[i].cpu().squeeze())
                    hm = output[i].detach().cpu().numpy().squeeze()
                    trgt = target[i].detach().cpu().numpy().squeeze()
                    # print(img_r.shape)
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_r)
                    # plt.plot(target.cpu().squeeze())
                    plt.plot(hm, alpha=0.3)
                    plt.subplot(1, 3, 2)
                    plt.imshow(trgt)
                    plt.subplot(1, 3, 3)
                    plt.imshow(hm)
                    plt.show()

    print('valid >>> epoch: {}/{}, mean_loss: {:.4f}'.format(
            epoch,
            epoch_range[-1],
            mean_loss
        ))
    loss_valid.append(mean_loss)

    log = {
        'loss_train':loss_train,
        'loss_valid':loss_valid
        }
    with open(filepath_hist, 'wb') as pfile:
        pickle.dump(log, pfile)
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

print(TIME_STAMP)





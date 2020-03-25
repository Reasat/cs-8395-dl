import os
import glob

data_root = r'D:\Data\cs-8395-dl\assignment3'
dir_train = os.path.join(data_root,'Training','img')
dir_test = os.path.join(data_root,'Testing','img')
dir_partition = r'D:\Projects\cs-8395-dl\Assignment_3_segmentation\partition\train_test_org'
path_test = os.path.join(dir_partition,'Testing.txt')
path_train = os.path.join(dir_partition,'Training.txt')
os.makedirs(dir_partition, exist_ok=True)

files_train = []
for dir, sub_dir, files in os.walk(dir_train):
    files_train+=files
print(files_train)
print(len(files_train))
with open(path_train,'w') as f:
    for file in files_train:
        f.write(file.replace('img','').replace('.nii.gz','')+'\n')

files_test=[]
for dir, sub_dir, files in os.walk(dir_test):
    files_test+=files
print(files_test)
print(len(files_test))
with open(path_test,'w') as f:
    for file in files_test:
        f.write(file.replace('img','').replace('.nii.gz','')+'\n')


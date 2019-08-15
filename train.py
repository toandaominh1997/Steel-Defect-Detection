import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.dataset import Steel
from datasets.augment import train_transforms
from datasets.augment import test_transforms

root = '/home/bigkizd/code/Steel-Defect-Detection/data'
path = '/home/bigkizd/code/Steel-Defect-Detection/data/train.csv'
batch_size = 16
df = pd.read_csv(path)

df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
df['ClassId']  = df['ClassId'].astype(int)
df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
df['defects'] = df.count(axis=1)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"])

train_dataset = Steel(root=root, state_root='images_train', data=train_df, transform=train_transforms)
val_dataset = Steel(root=root, state_root='images_train', data=val_df, transform=test_transforms)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

image, mask =  train_dataset[0]
image.save('image.png')
mask.save('mask.png')



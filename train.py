import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.dataset import Steel
from datasets.augment import train_transforms
from datasets.augment import test_transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.model import UNet
from tqdm import tqdm
root = '/home/bigkizd/code/Steel-Defect-Detection/data'
path = '/home/bigkizd/code/Steel-Defect-Detection/data/train.csv'
batch_size = 16
df = pd.read_csv(path)

df = df[df['EncodedPixels'].notnull()].reset_index(drop=True)
df = df[df['ImageId_ClassId'].apply(lambda x: x.split('_')[1]=='4')].reset_index(drop=True)


train_df, val_df = train_test_split(df, test_size=0.2)
train_transform = train_transforms()
test_transform = test_transforms()
train_dataset = Steel(root=root, state_root='train_images', data=train_df)
val_dataset = Steel(root=root, state_root='train_images', data=val_df)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# # print('image: ', image.size())
model = UNet(n_channels=3, n_classes=4)
criterion = nn.BCEWithLogitsLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr = 0.001, momentum=0.9)


def train(data_loader):   
    model.train()
    for idx, (images, targets) in enumerate(tqdm(data_loader)):
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    return loss    

loss = train(train_dataloader)

print(loss)


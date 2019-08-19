from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from PIL import Image
import os 
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.torch import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt


def get_transforms():
    transform = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.ToTensor()
    ])
    return transform

class Steel(Dataset):
    def __init__(self, root, state_root, data, mean=None, std=None, phase=None):
        super(Steel, self).__init__()
        self.root = root
        self.state_root = state_root 
        self.df = data
        self.transform = get_transforms() 

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        try:
            mask = self.rle2mask(self.df['EncodedPixels'].iloc[index], (256, 1600), index)

            image = Image.open(os.path.join(self.root, self.state_root, self.df.iloc[index]['ImageId_ClassId'].split('_')[0]))
        except IOError:
            print('Corrupted image for {}'.format(index))
            return self[index+1]

        mask = transforms.ToPILImage()(mask)
        image = self.transform(image)
        mask = self.transform(mask)
        
        return image, mask

    def make_mask(self, index, df):
        fname = df.iloc[index].name
        labels = df.iloc[index, :4]

        masks = np.zeros((256, 1600, 4), dtype=np.float32)
        count = 0 
        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                print('index:', idx)
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(256*1600, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos+le)] = 1
                masks[:, :, idx] = mask.reshape(256, 1600, order='F')
                count = 1
        print('kaka')
        return fname, masks, count
    def rle2mask(self, rle, imgshape, row_id):
        width = imgshape[0]
        height= imgshape[1]
        masks = np.zeros((height, width, 4), dtype=np.uint8)
        mask= np.zeros( width*height ).astype(np.uint8)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0

        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]
        label = self.df.iloc[row_id]['ImageId_ClassId'].split('_')[1]
        label = int(label)-1
        masks[:, :, label] = mask.reshape(height, width, order='F')
        return masks




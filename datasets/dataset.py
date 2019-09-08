import os 
import cv2
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split 
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    Normalize 
)
from albumentations.torch import ToTensor
from torchvision import transforms 
import random 
from imgaug import augmenters as iaa


def get_transforms(phase):
    original_height = 256
    original_width = 1600
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
            OneOf([
                RandomSizedCrop(min_max_height=(50, 101), height=original_height, width=original_width, p=0.5),
                PadIfNeeded(min_height=original_height, min_width=original_width, p=0.5)], p=1),    
                VerticalFlip(p=0.5),              
                # RandomRotate90(p=0.5),
                OneOf([
                    ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    GridDistortion(p=0.5),
                    OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
                ], p=0.8),
                CLAHE(p=0.8),
                RandomBrightnessContrast(p=0.8),    
                RandomGamma(p=0.8),
            ]
        )
    list_transforms.extend(
        [
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
class SteelDataset(Dataset):
    def __init__(self, root_dataset, list_data, phase):
        super(SteelDataset, self).__init__()
        self.root_dataset = root_dataset
        self.df = self.__read_file__(list_data=list_data)
        self.transforms = get_transforms(phase)
    
    def __read_file__(self, list_data):
        df = pd.read_csv(os.path.join(list_data))
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        # df, non_df = train_test_split(df, test_size=0.98) 
        return df
    def make_mask(self, row_id, df):
        fname = df.iloc[row_id].name
        labels = df.iloc[row_id][:4]
        masks = np.zeros((256, 1600, 4), dtype=np.float32)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(256 * 1600, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos + le)] = 1
                masks[:, :, idx] = mask.reshape(256, 1600, order='F')
        return fname, masks
    def __getitem__(self, index):
        image_id, mask = self.make_mask(index, self.df)
        image_path = os.path.join(self.root_dataset, image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 1x4x256x1600
        return img, mask

    def __len__(self):
        return len(self.df)
    


import os 
import cv2
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split 
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

from torchvision import transforms 
import random 


def get_transforms(phase, width=1600, height=256):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                    ], p=0.3),
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                albu.ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.Resize(width,height,always_apply=True),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )
    list_trfms = albu.Compose(list_transforms)
    return list_trfms
class SteelDataset(Dataset):
    def __init__(self, root_dataset, list_data, phase, mode='cls'):
        super(SteelDataset, self).__init__()
        self.mode = mode
        self.root_dataset = root_dataset
        self.df = self.__read_file__(list_data=list_data)
        self.transforms = get_transforms(phase, width = 1600, height = 800)
    
    def __read_file__(self, list_data):
        df = pd.read_csv(os.path.join(list_data))
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
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
        # if self.mode == 'cls':
        #     mask = mask[0].permute(2, 0, 1) # 1x4x256x1600
        #     mask = (mask.view(4, -1).sum(1)>0)
        #     mask = mask.float()
        # else:
        #     mask = mask*torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        #     mask, _ = torch.max(mask, -1) 
        #     mask = mask.long()
        return img, mask

    def __len__(self):
        return len(self.df)
        return 30
    


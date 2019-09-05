import os 
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
class SteelDataset(Dataset):
    def __init__(self, root_dataset, list_data):
        super(SteelDataset, self).__init__()
        self.root_dataset = root_dataset
        self.df = self.__read_file__(list_data=list_data)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    def __read_file__(self, list_data):
        df = pd.read_csv(os.path.join(list_data))
        df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
        df['ClassId'] = df['ClassId'].astype(int)
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        df, non_df = train_test_split(df, test_size=0.95)
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
        image_id, segm = self.make_mask(index, self.df)
        image_path = os.path.join(self.root_dataset, image_id)
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        segm = self.segm_transform(segm)
        return image, segm

    def __len__(self):
        return len(self.df)
    
    def image_transform(self, image):
        image = np.float32(np.array(image))
        image = image.transpose((2, 0, 1))
        image = self.normalize(torch.from_numpy(image.copy()))
        return image 
    def segm_transform(self, segm):
        segm = torch.from_numpy(np.array(segm)).float()
        segm = segm.permute(2, 0, 1)
        return segm 


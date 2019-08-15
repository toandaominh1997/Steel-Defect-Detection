from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
from PIL import Image
import os 

class Steel(Dataset):
    def __init__(self, root, state_root, data, transform):
        super(Steel, self).__init__()
        self.root = root
        self.state_root = state_root
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        try:
            image_id, mask = self.make_mask(index, self.data)
            image = Image.open(os.path.join(self.root, self.state_root, image_id))
        except IOError:
            print('Corrupted image for {}'.format(index))
            return self[index+1]
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = mask[0].permute(2, 0, 1)
        return image, mask

    def make_mask(self, index, df):
        fname = df.iloc[index].name
        labels = df.iloc[index][:4]
        
        masks = np.zeros((256, 1600, 4), dtype=np.float32)
        for idx, label in enumerate(labels.values):
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256*1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos+le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')
        return fname,masks



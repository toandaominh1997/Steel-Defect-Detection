import os 
os.system('pip install -r requirements.txt')
import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets.augment import train_transforms
from datasets.augment import test_transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.unet import Unet
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch.optim as optim
import time

from datasets.dataset import SteelDataset
from utils.metric import *
import torch.backends.cudnn as cudnn
seed = 69
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def provider(data_folder, df_path, phase, mean=None, std=None, batch_size=16, num_workers=8):
    df = pd.read_csv(df_path)
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    df['defects'] = df.count(axis=1)
    train_df, val_df = train_test_split(df, test_size=0.02, stratify=df["defects"])
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(image_dataset,batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    return dataloader

model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
#         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tqdm(dataloader)): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
#             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "/opt/ml/model/model.pth")
            print()

sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './data/train.csv'
data_folder = "./data"
test_data_folder = "./train_images"
model_trainer = Trainer(model)
model_trainer.start()

# model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
# criterion = nn.BCEWithLogitsLoss()
# if torch.cuda.is_available():
#     model = model.cuda()
#     criterion = criterion.cuda()
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-4, lr = 5e-4, momentum=0.9)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model)
# def train(data_loader):   
#     model.train()
#     for idx, (images, targets) in enumerate(tqdm(data_loader)):
#         images = images.to(device)
#         targets = targets.to(device)
#         outputs = model(images)
#         optimizer.zero_grad()
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#     return loss    
# def evaluate(data_loader):
#     model.eval()
#     meter = Meter()
#     for idx, (images, targets) in enumerate(tqdm(data_loader)):
#         images = images.to(device)
#         targets = targets.to(device)
#         outputs = model(images)
#         loss = criterion(outputs, targets)
#         targets= targets.cpu().detach()
#         outputs = outputs.cpu().detach()
#         meter.update(targets, outputs)
#     dices, iou = meter.get_metrics()
#     dice, dice_neg, dice_pos = dices
#     print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (loss, iou, dice, dice_neg, dice_pos))
#     return loss



# def main():
#     for epoch in range(num_epoch):
#         train(train_loader)
#         evaluate(val_loader)

# if __name__ == '__main__':
#     main() 

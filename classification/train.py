import os 
os.system('pip install -r requirements.txt')
import argparse 
import torch 
from datasets import SteelDataset 
from torch.utils.data import DataLoader 
import torch.nn as nn 
from optimizers import RAdam 
from utils import Meter 
from tqdm import tqdm 
import torch.optim as optim
from modules import segmentation_models as smp
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modelss import ModelBuilder
import torchvision


parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--root_dataset', default='./data/train_images', type=str, help='config file path (default: None)')
parser.add_argument('--list_train', default='./data/train.csv', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_class', default=4, type=int)

args = parser.parse_args()
import os 
os.system('pip install -r requirements.txt')
import argparse 
import torch 
from datasets import SteelDataset 
from torch.utils.data import DataLoader 
import torch.nn as nn 
from optimizers import RAdam 
from utils import Metric
from tqdm import tqdm 
import torch.optim as optim
from models.model import Model
import time 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from models.loss import Criterion
from datasets.dataset import null_collate


parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--root_dataset', default='./data/train_images', type=str, help='config file path (default: None)')
parser.add_argument('--list_train', default='./data/train.csv', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_class', default=5, type=int)

parser.add_argument('--encoder', default="resnet34", type=str)
parser.add_argument('--decoder', default="hrnet", type=str)  
parser.add_argument('--encoder_weights', default="imagenet", type=str) 
parser.add_argument('--mode', default='non-cls', type=str)
args = parser.parse_args()

if args.mode == 'cls':
    arch='classification'
    args.num_class = 4
else:
    args.num_class = 5
    arch = '{}_{}_{}'.format(args.mode, args.encoder, args.decoder)
print('Architectyre: {}'.format(arch))

train_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='train', mode=args.mode)
valid_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='valid', mode=args.mode)


model = Model(num_class=args.num_class, encoder = args.encoder, decoder = args.decoder, mode=args.mode)
model = model.cuda()
criterion = Criterion(mode=args.mode)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

def choosebatchsize(dataset, model, optimizer, criterion):
    batch_size = 33
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory = True)
    dataloader_iterator = iter(data_loader)
    model = model.cuda()
    model.train()
    while True:
        try:
            image, target = next(dataloader_iterator)
            image = image.cuda()
            target = target.cuda() 
            outputs = model(image) 
            loss = criterion(outputs, target) 
            loss.backward() 
            optimizer.zero_grad() 
            optimizer.step() 
            image = None 
            target = None 
            outputs = None  
            loss = None
            torch.cuda.empty_cache() 
            return batch_size 
        except RuntimeError as e: 
            print('Runtime Error {} at batch size: {}'.format(e, batch_size)) 
            batch_size = batch_size - 2
            if batch_size<=0:
                batch_size = 1
            data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory = True) 
            dataloader_iterator = iter(data_loader) 

args.batch_size = choosebatchsize(train_dataset, model, optimizer, criterion)
args.batch_size = args.batch_size - 1
if args.batch_size < 1:
    args.batch_size = 1
print('Choose batch_size: ', args.batch_size)


train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4, pin_memory = True)
valid_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=4, pin_memory = True)

def train(data_loader):
    model.train()
    total_loss = 0
    accumulation_steps = 32 // args.batch_size
    optimizer.zero_grad()
    for idx, (img, segm) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        segm = segm.cuda()
        outputs = model(img)
        loss = criterion(outputs, segm)
        (loss/accumulation_steps).backward()
        clipping_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        if (idx + 1 ) % accumulation_steps == 0:
            optimizer.step() 
            optimizer.zero_grad() 
        total_loss += loss.item() 
    torch.cuda.empty_cache()
    return total_loss/len(data_loader)

def evaluate(data_loader):
    meter = Metric(mode=args.mode)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (img, segm) in enumerate(data_loader):
            img = img.cuda() 
            segm = segm.cuda() 
            outputs = model(img) 
            loss = criterion(outputs, segm)
            outputs = outputs.detach().cpu()
            segm = segm.detach().cpu() 
            meter.update(segm, outputs) 
            total_loss += loss.item()
        if args.mode == 'cls':
            tn, tp = meter.get_metrics() 
            return total_loss/len(data_loader), tn, tp 
        else:
            dices, iou = meter.get_metrics() 
            dice, dice_neg, dice_pos = dices 
            torch.cuda.empty_cache() 
            return total_loss/len(data_loader), iou, dice, dice_neg, dice_pos


best_loss = float("inf")
for epoch in range(args.num_epoch):
    start_time = time.time()
    loss_train = train(train_loader)
    print('[TRAIN] Epoch: {}| Loss: {}| Time: {}'.format(epoch, loss_train, time.time()-start_time))
    if (epoch+1)%5==0:
        start_time = time.time()
        if args.mode == 'cls':
            val_loss, tn, tp = evaluate(valid_loader)
            print("['EVAL'] Epoch: {}|Loss: {}| tn: {}| tp: {}| time: {}".format(epoch, val_loss, tn, tp, time.time()-start_time))
        else:
            val_loss, iou, dice, dice_neg, dice_pos = evaluate(valid_loader)
            print("['EVAL'] Epoch: {}|Loss: {}| IoU: {}| dice: {}| dice_neg: {}| dice_pos: {}| time: {}".format(epoch, val_loss, iou, dice, dice_neg, dice_pos, time.time()-start_time))
        scheduler.step(val_loss)
        
        if val_loss < best_loss or (epoch+1)%10==0:
            status = "not best loss"
            if val_loss < best_loss:
                status = "best loss"
                best_loss = val_loss
                        
            state = {
                "status": status,
                "epoch": epoch,
                "arch": arch,
                "state_dict": model.state_dict()
            }
            torch.save(state, '/opt/ml/model/{}_checkpoint_{}.pth'.format(arch, epoch))

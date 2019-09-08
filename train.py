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

parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--root_dataset', default='./data/train_images', type=str, help='config file path (default: None)')
parser.add_argument('--list_train', default='./data/train.csv', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--encoder', default="resnet50", type=str)
parser.add_argument('--decoder', default="Unet", type=str)  
parser.add_argument('--encoder_weights', default="imagenet", type=str) 
args = parser.parse_args()

print('Encoder: {}, Decoder: {}'.format(args.encoder, args.decoder))

# net_encoder = ModelBuilder.build_encoder(
#         arch="hrnetv2",
#         fc_dim=720,
#         weights='')
# net_decoder = ModelBuilder.build_decoder(
#     arch="c1",
#     fc_dim=720,
#     num_class=4,
#     weights='')


modules = {
    "{}_Unet".format(args.encoder): smp.Unet('{}'.format(args.encoder), classes=args.num_class, activation='softmax', encoder_weights=args.encoder_weights),
    "{}_Linknet".format(args.encoder): smp.Linknet('{}'.format(args.encoder), classes=args.num_class, activation='softmax', encoder_weights=args.encoder_weights),
    "{}_FPN".format(args.encoder): smp.FPN('{}'.format(args.encoder), classes=args.num_class, activation='softmax', encoder_weights=args.encoder_weights),
    "{}_PSPNet".format(args.encoder): smp.PSPNet('{}'.format(args.encoder), classes=args.num_class, activation='softmax', encoder_weights=args.encoder_weights),
    # "hrnetv2_c1": nn.Sequential(net_encoder, net_decoder)

}
train_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='train')
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)

valid_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='valid')
valid_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)


criterion = nn.BCEWithLogitsLoss()
model = modules["{}_{}".format(args.fencoder, args.fdecoder)]
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

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
        loss.backward()
        if (idx + 1 ) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss
    return total_loss/len(data_loader)

def evaluate(data_loader):
    meter = Meter('eval', 0)
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
            total_loss += loss
        dices, iou = meter.get_metrics() 
        dice, dice_neg, dice_pos = dices
        
        return total_loss/len(data_loader), iou, dice, dice_neg, dice_pos


best_loss = float("inf")
for epoch in range(args.num_epoch):
    start_time = time.time()
    loss_train = train(train_loader)
    print('[TRAIN] Epoch: {}| Loss: {}| Time: {}'.format(epoch, loss_train, time.time()-start_time))
    if (epoch+1)%5==0:
        start_time = time.time()
        val_loss, iou, dice, dice_neg, dice_pos = evaluate(valid_loader)
        scheduler.step(val_loss)
        print("['EVAL'] Epoch: {}|Loss: {}| IoU: {}| dice: {}| dice_neg: {}| dice_pos: {}| time: {}".format(epoch, val_loss, iou, dice, dice_neg, dice_pos, time.time()-start_time))
        if val_loss < best_loss or (epoch+1)%10==0:
            status = "not best loss"
            if val_loss < best_loss:
                status = "best loss"
            best_loss = val_loss
            state = {
                "status": status,
                "epoch": epoch,
                "arch": "{}_{}".format(args.encoder, args.decoder),
                "state_dict": model.state_dict(),
            }
            torch.save(state, '/opt/ml/model/{}_{}_checkpoint_{}.pth'.format(args.encoder, args.decoder, epoch))

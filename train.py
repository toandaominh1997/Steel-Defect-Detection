import argparse 
import torch 
from datasets import SteelDataset 
from torch.utils.data import DataLoader 
import torch.nn as nn 
from optimizers import RAdam 
from utils import Meter 
from tqdm import tqdm 
from modules import segmentation_models as smp


def train(data_loader, model, criterion, optimizer):
    model.train()
    for idx, (img, segm) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        segm = segm.cuda()
        outputs = model(img)
        loss = criterion(outputs, segm)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
    return loss/len(data_loader)

def evaluate(epoch, data_loader, model, criterion):
    meter = Meter('eval', epoch)
    model.eval()
    with torch.no_grad():
        for idx, (img, segm) in enumerate(data_loader):
            img = img.cuda() 
            segm = segm.cuda() 
            outputs = model(img) 
            loss = criterion(outputs, segm)
            outputs = outputs.detach().cpu()
            segm = segm.detach().cpu() 
            meter.update(segm, outputs) 
        dices, iou = meter.get_metrics() 
        dice, dice_neg, dice_pos = dices
        print("['EVAL'] Epoch: {}|Loss: {}| IoU: {}| dice: {}| dice_neg: {}| dice_pos: {}".format(epoch, loss, iou, dice, dice_neg, dice_pos))
        return loss



def main(args):
    modules = {
        "Unet_Resnet34": smp.Unet('resnet34', classes=args.num_class, activation='softmax'),
        "Linknet_Resnet34": smp.Linknet('resnet34', classes=args.num_class, activation='softmax'),
        "FPN_Resnet34": smp.Unet('resnet34', classes=args.num_class, activation='softmax'),
        "PSPNet_Resnet34": smp.PSPNet('resnet34', classes=args.num_class, activation='softmax'),
    }
    train_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4)

    criterion = nn.BCEWithLogitsLoss()
    # model = Unet("resnet50", encoder_weights="imagenet", classes=4, activation=None)
    model = modules[args.module_name]
    model = model.cuda()
    optimizer = RAdam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    for epoch in range(args.num_epoch):
        loss_train = train(train_loader, model, criterion, optimizer)
        print('[TRAIN] Epoch: {}| Loss: {}'.format(epoch, loss_train))
        val_loss = evaluate(epoch, train_loader, model, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            state = {
                "epoch": epoch,
                "arch": model.__class__.__name__,
                "state_dict": model.state_dict(),
            }

            torch.save(state, '/opt/ml/model/{}_checkpoint_{}.pth'.format(args.module_name, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--root_dataset', default='./data/train_images', type=str, help='config file path (default: None)')
    parser.add_argument('--list_train', default='./data/train.csv', type=str)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--module_name', default="PSPNet_Resnet34", type=str)
    args = parser.parse_args()

    main(args)
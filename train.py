import os 
# os.system('pip install -r requirements.txt')
import argparse 
import time 
from pathlib import Path
from learning import Learning
from utils import load_yaml
import importlib
import torch

def getattribute(config, name_package, *args, **kwargs):
    module = importlib.import_module(config[name_package]['PY'])
    module_class = getattr(module, config[name_package]['CLASS'])
    module_args = dict(config[name_package]['ARGS'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    package = module_class(*args, **module_args)
    return package

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser.add_argument('--train_cfg', type=str, default='./configs/train_config.yaml', help='train config path')
    args = parser.parse_args()
    config_folder = Path(args.train_cfg.strip("/"))
    train_config = load_yaml(config_folder)
    
    train_dataset = getattribute(config = train_config, name_package = 'TRAIN_DATASET')
    valid_dataset = getattribute(config = train_config, name_package = 'VALID_DATASET')
    train_dataloader = getattribute(config = train_config, name_package = 'TRAIN_DATALOADER', dataset = train_dataset)
    valid_dataloader = getattribute(config = train_config, name_package = 'VALID_DATALOADER', dataset = valid_dataset)
    model = getattribute(config = train_config, name_package = 'MODEL')
    criterion = getattribute(config = train_config, name_package = 'CRITERION')
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattribute(config = train_config, name_package= 'OPTIMIZER', params = model.parameters())
    scheduler = getattribute(config = train_config, name_package = 'SCHEDULER', optimizer = optimizer)
    num_epoch = train_config['NUM_EPOCH']
    gradient_clipping = train_config['GRADIENT_CLIPPING']
    gradient_accumulation_steps = train_config['GRADIENT_ACCUMULATION_STEPS']
    early_stopping = train_config['EARLY_STOPPING']
    validation_frequency = train_config['VALIDATION_FREQUENCY']
    saved_period = train_config['SAVED_PERIOD']
    checkpoint_dir = train_config['CHECKPOINT_DIR']
    resume_path = train_config['RESUME_PATH']
    learning = Learning(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=torch.device('cuda:0'),
                        num_epoch=num_epoch,
                        scheduler = scheduler,
                        grad_clipping = gradient_clipping,
                        grad_accumulation = gradient_accumulation_steps,
                        early_stopping = early_stopping,
                        validation_frequency = validation_frequency,
                        save_period = saved_period,
                        checkpoint_dir = checkpoint_dir,
                        resume_path=resume_path)
    learning.train(train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()





































# parser = argparse.ArgumentParser(description='Semantic Segmentation')
# parser.add_argument('--root_dataset', default='./data/train_images', type=str, help='config file path (default: None)')
# parser.add_argument('--list_train', default='./data/train.csv', type=str)
# parser.add_argument('--batch_size', default=16, type=int)
# parser.add_argument('--lr', default=5e-4, type=float)
# parser.add_argument('--num_epoch', default=200, type=int)
# parser.add_argument('--num_class', default=4, type=int)
# parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
# parser.add_argument('--encoder', default="resnet34", type=str)
# parser.add_argument('--decoder', default="Unet", type=str)  
# parser.add_argument('--encoder_weights', default="imagenet", type=str) 
# parser.add_argument('--mode', default='non-cls', type=str)

# parser.add_argument('train_cfg', type=str, help='train config path')
# args = parser.parse_args()

# config_folder = Path(args.train_cfg.strip("/"))

# if args.mode == 'cls':
#     arch='classification'
#     args.num_class = 4
# else:
#     args.num_class = 4
#     arch = '{}_{}_{}'.format(args.mode, args.encoder, args.decoder)
# print('Architectyre: {}'.format(arch))

# train_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='train', mode=args.mode)
# valid_dataset = SteelDataset(root_dataset = args.root_dataset, list_data = args.list_train, phase='valid', mode=args.mode)


# model = Model(num_class=args.num_class, encoder = args.encoder, decoder = args.decoder, mode=args.mode)
# model = model.cuda()
# criterion = Criterion(mode=args.mode)
# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)

# def choosebatchsize(dataset, model, optimizer, criterion):
#     batch_size = 33
#     data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory = True)
#     dataloader_iterator = iter(data_loader)
#     model = model.cuda()
#     model.train()
#     while True:
#         try:
#             image, target = next(dataloader_iterator)
#             image = image.cuda()
#             target = target.cuda() 
#             outputs = model(image) 
#             loss = criterion(outputs, target) 
#             loss.backward() 
#             optimizer.zero_grad() 
#             optimizer.step() 
#             image = None 
#             target = None 
#             outputs = None  
#             loss = None
#             torch.cuda.empty_cache() 
#             return batch_size 
#         except RuntimeError as e: 
#             print('Runtime Error {} at batch size: {}'.format(e, batch_size)) 
#             batch_size = batch_size - 2
#             if batch_size<=0:
#                 batch_size = 1
#             data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory = True) 
#             dataloader_iterator = iter(data_loader) 

# # args.batch_size = choosebatchsize(train_dataset, model, optimizer, criterion)
# # args.batch_size = args.batch_size - 1
# # if args.batch_size < 1:
# #     args.batch_size = 1
# args.batch_size = 2
# print('Choose batch_size: ', args.batch_size)

# train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=4, pin_memory = True)
# valid_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False, num_workers=4, pin_memory = True)

# learning = Learning(model=model,
#         optimizer=optimizer,
#         criterion=criterion,
#         device=torch.device('cuda:0'),
#         n_epoches=50,
#         scheduler = scheduler,
#         grad_clipping = 1.0,
#         grad_accumulation = 1.0,
#         early_stopping = 10,
#         validation_frequency = 5,
#         save_period = 5,
#         checkpoint_dir = './saved/',
#         resume_path=None)

# learning.train(train_loader, valid_loader)
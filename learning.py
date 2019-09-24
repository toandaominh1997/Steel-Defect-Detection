from tqdm import tqdm
import torch
from utils import AverageMetric
import os
class Learning(object):
    def __init__(self,
            model,
            optimizer,
            criterion,
            device,
            num_epoch,
            scheduler,
            grad_clipping,
            grad_accumulation,
            early_stopping,
            validation_frequency,
            save_period,
            checkpoint_dir,
            resume_path):
        self.device, device_ids = self._prepare_device(device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch 
        self.scheduler = scheduler
        self.grad_clipping = grad_clipping
        self.grad_accumulation = grad_accumulation
        self.early_stopping = early_stopping
        self.validation_frequency =validation_frequency
        self.save_period = save_period
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_score = 0
        if resume_path is not None:
            self._resume_checkpoint(resume_path)
        
    def train(self, train_dataloader, valid_dataloader):
        for epoch in range(self.start_epoch, self.num_epoch+1):
            print("{} epoch: \t start training....".format(epoch))
            train_loss_mean = self._train_epoch(train_dataloader)
            
            print("{} epoch: \t Calculated train loss: {:5}".format(epoch, train_loss_mean))

            if (epoch+1) % self.validation_frequency==0:
                print("skip validation....")
                continue
            print('{} epoch: \t start validation....'.format(epoch))
            valid_loss_mean, dice_score = self._valid_epoch(valid_dataloader)
            print("{} epoch: \t Calculated valid loss: {:5} \t dice score: {:5}".format(epoch, train_loss_mean, dice_score))
            self.post_processing(dice_score, epoch)
            if epoch - self.best_epoch > self.early_stopping:
                print('EARLY STOPPING')
                break
    def _train_epoch(self, loader):
        self.model.train()
        self.optimizer.zero_grad()
        current_loss_mean = 0.0
        for batch_idx, (batch_imgs, batch_labels) in enumerate(tqdm(loader)):
            batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
            outputs = self.model(batch_imgs)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            if (batch_idx+1) % self.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clipping)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            current_loss_mean += loss.item()
        return current_loss_mean/len(loader)
    
    def _valid_epoch(self, loader):
        metrics = AverageMetric()
        self.model.eval()
        current_loss_mean = 0 
        metrics.reset()
        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels) in enumerate(tqdm(loader)):
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
                outputs = self.model(batch_imgs)
                loss = self.criterion(outputs, batch_labels)
                current_loss_mean += loss.item()
                metrics.update(outputs, batch_labels)
            return current_loss_mean/len(loader), metrics.value()
    def post_processing(self, score, epoch):
        best = False
        if score > self.best_score:
            self.best_score = score 
            self.best_epoch = epoch 
            best = True
            print("best model: {} epoch - {:.5}".format(epoch, score))
        if best==True or (self.save_period>=0 and epoch % self.save_period == 0):
            self._save_checkpoint(epoch = epoch, save_best = best)
        
        if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            self.scheduler.step(score)
        else:
            self.scheduler.step()
    
    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.get_state_dict(self.model),
            'best_score': self.best_score
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict
    
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
    def _prepare_device(self, device):
        if type(device)==int:
            n_gpu_use = device
        else:
            n_gpu_use = len(device)
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if type(device)==int:
            device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
            list_ids = list(range(n_gpu_use))
        elif len(device) == 1:
            list_ids = device
            if device[0] >= 0 and device[0] < n_gpu:    
                device = torch.device('cuda:{}'.format(device[0]))
            else:
                device = torch.device('cuda:0')
        else:
            list_ids = device
            device = torch.device('cuda:{}'.format(device[0]) if n_gpu_use > 0 else 'cpu')
            
        return device, list_ids

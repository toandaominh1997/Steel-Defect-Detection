import torch
import torch.nn as nn 
import torchvision 
import pretrainedmodels
import torch.nn.functional as F


class Resnet34Classification(nn.Module):
    def __init__(self, num_class):
        super(Resnet34Classification, self).__init__()
        models = pretrainedmodels.resnet34()
        self.resnet = nn.Sequential(*list(models.children())[:-2])
        self.feature = nn.Conv2d(512, 32, kernel_size=1)
        self.out = nn.Conv2d(32, num_class, kernel_size=1)
    def forward(self, input):
        out = self.resnet(input)
        out = F.dropout(out,0.5)
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.feature(out)
        out = self.out(out)
        return out

def criterion_cls(logit, truth, weight=None):
    batch_size, num_class, H, W = logit.shape 
    logit = logit.view(batch_size, num_class)
    truth = truth.view(batch_size, num_class)
    assert(logit.shape==truth.shape)
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if weight is None:
        loss = loss.mean()
    else:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_sum = pos.sum().item() + 1e-12
        neg_sum = neg.sum().item() + 1e-12
        loss = (weight[1]*pos*loss/pos_sum + weight[0]*neg*loss/neg_sum).sum()
    return loss 

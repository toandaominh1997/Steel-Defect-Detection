import torch 
from models.model import Model


inputs = torch.randn(2, 3, 256, 1600)

# model = Model(mode='non-cls',
#     encoder='resnet34',
#     decoder='hrnet')


# outputs= model(inputs)

# print(outputs.size())

import numpy as np

def metric_hit(logit, truth, threshold=0.5):
    batch_size,num_class, H,W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size,num_class,-1)
        truth = truth.view(batch_size,num_class,-1)

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives

        
        tp = tp.sum(dim=[0,2])
        tn = tn.sum(dim=[0,2])
        num_pos = t.sum(dim=[0,2]) 
        num_neg = batch_size*H*W - num_pos

        tp = tp.data.cpu().numpy()
        tn = tn.data.cpu().numpy().sum()
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().sum()
        print('tn: ', tn)
        print('tp: ', tp)
        
        tp = np.nan_to_num(tp/(num_pos+1e-12),0)
        tn = np.nan_to_num(tn/(num_neg+1e-12),0)


        tp = list(tp)
        num_pos = list(num_pos)

    return tn,tp, num_neg,num_pos
from utils.meter import Metric


inputs = torch.randn(2, 4, 1, 1)
outputs = torch.randn(2, 4, 1, 1)
tn, tp, num_neg, num_pos = metric_hit(inputs, inputs)
metric = Metric(mode='cls')
metric.update(inputs, inputs)
tn, tp = metric.get_metrics()
print(tn, tp)
import torch.nn.modules.loss as loss
import torch.nn.functional as F
import torch 

class Criterion(loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', mode='cls'):
        super(Criterion, self).__init__(size_average, reduce, reduction)
        self.mode = mode 
    def forward(self, logit, target, weight=None):
        loss = 0
        if self.mode =='cls':
            batch_size,num_class, H,W = logit.shape
            logit = logit.view(batch_size,num_class)
            truth = truth.view(batch_size,num_class)
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
        else:
            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
            truth = truth.permute(0, 2, 3, 1).contiguous().view(-1)
            
            if weight is not None: weight = torch.FloatTensor([1]+weight).cuda()
            loss = F.cross_entropy(logit, truth, weight=weight, reduction='none')

            loss = loss.mean()
            return loss
        return loss
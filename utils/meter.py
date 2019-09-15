import numpy as np 
import torch 

class Metric(object):
    def __init__(self, mode):
        super(Metric, self).__init__()
        self.mode = mode
        self.base_threshold = 0.5 
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []
        self.tn = []
        self.tp = []

    def update(self, targets, outputs):
        if self.mode =='cls':

            batch_size,num_class, H,W = outputs.shape

            with torch.no_grad():
                outputs = outputs.view(batch_size,num_class,-1)
                targets = targets.view(batch_size,num_class,-1)

                probability = torch.sigmoid(outputs)
                p = (probability>self.base_threshold).float()
                t = (targets>0.5).float()

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

                tp = np.nan_to_num(tp/(num_pos+1e-12),0)
                tn = np.nan_to_num(tn/(num_neg+1e-12),0)

                tp = list(tp)
                num_pos = list(num_pos)
            
            self.tn.append(tn*num_neg)
            self.tp.append(list(np.array(tp)*np.array(num_pos)))
        else:
            probs = torch.sigmoid(outputs)
            dice, dice_neg, dice_pos, _, _ = self.metric(probs, targets, self.base_threshold)
            self.base_dice_scores.append(dice)
            self.dice_pos_scores.append(dice_pos)
            self.dice_neg_scores.append(dice_neg)
            preds = self.predict(probs, self.base_threshold)
            iou = self.compute_iou_batch(preds, targets, classes=[1])
            self.iou_scores.append(iou)

    def get_metrics(self):
        if self.mode =='cls':
            tn= np.mean(self.tn)
            tp = np.mean(self.tp, axis=0)
            return tn, tp
        else:
            dice = np.mean(self.base_dice_scores)
            dice_neg = np.mean(self.dice_neg_scores)
            dice_pos = np.mean(self.dice_pos_scores)
            dices = [dice, dice_neg, dice_pos]
            iou = np.nanmean(self.iou_scores)
            return dices, iou
    def predict(self, X, threshold):
        '''X is sigmoid output of the model'''
        X_p = np.copy(X)
        preds = (X_p > threshold).astype('uint8')
        return preds
    def metric(self, probability, truth, threshold=0.5, reduction='none'):
        '''Calculates dice of positive and negative images seperately'''
        '''probability and truth must be torch tensors'''
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert(probability.shape == truth.shape)

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

            dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
            dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
            dice = dice.mean().item()

            num_neg = len(neg_index)
            num_pos = len(pos_index)

        return dice, dice_neg, dice_pos, num_neg, num_pos
    def compute_iou_batch(self, outputs, labels, classes=None):
        '''computes mean iou for a batch of ground truth masks and predicted masks'''
        ious = []
        preds = np.copy(outputs) # copy is imp
        labels = np.array(labels) # tensor to np
        for pred, label in zip(preds, labels):
            ious.append(np.nanmean(self.compute_ious(pred, label, classes)))
        iou = np.nanmean(ious)
        return iou

    def compute_ious(self, pred, label, classes, ignore_index=255, only_present=True):
        '''computes iou for one ground truth mask and predicted mask'''
        pred[label == ignore_index] = 0
        ious = []
        for c in classes:
            label_c = label == c
            if only_present and np.sum(label_c) == 0:
                ious.append(np.nan)
                continue
            pred_c = pred == c
            intersection = np.logical_and(pred_c, label_c).sum()
            union = np.logical_or(pred_c, label_c).sum()
            if union != 0:
                ious.append(intersection / union)
        return ious if ious else [1]


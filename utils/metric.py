import numpy as np 
import torch 

class AverageMetric(object):
    def __init__(self, threshold=0.5):
        self.dice_scores = []
        self.threshold = threshold
    def update(self, outputs, labels):
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            dice_score = self.dice_metric(probability=probs, truth = labels)
            self.dice_scores.append(dice_score)
    def value(self):
        return np.mean(self.dice_scores)
    def reset(self):
        self.dice_scores = []
    def dice_metric(self, probability, truth):
        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert(probability.shape == truth.shape) 
            p = (probability > self.threshold).float() 
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
            dice = dice.mean().item()
        return dice
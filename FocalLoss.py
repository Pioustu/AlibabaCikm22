# Reference:
# https://cloud.tencent.com/developer/article/1669261
# https://www.jianshu.com/p/0c159cdd9c50
# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha]).cuda()        
            self.gamma = gamma
            
    def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
            # targets = targets.type(torch.long)        
            # at = self.alpha.gather(0, targets.data.view(-1))        
            pt = torch.exp(-BCE_loss)
            F_loss = (1-pt)**self.gamma * BCE_loss        
            return F_loss.mean()
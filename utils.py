import logging
import random

import torch.backends.cudnn as cudnn
import torch
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


logger = logging.getLogger(__name__)

class CIKMCUPDataset(InMemoryDataset):
    name = 'CIKM22Competition'
    inmemory_data = {}

    def __init__(self, root):
        super(CIKMCUPDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        pass

    def __getitem__(self, idx):
        if idx in self.inmemory_data:
            return self.inmemory_data[idx]
        else:
            self.inmemory_data[idx] = {}
            for split in ['train', 'val', 'test']: 
                split_data = self._load(idx, split)
                if split_data:
                    self.inmemory_data[idx][split] = split_data
            return self.inmemory_data[idx]

def load_client_data(config,used_id=[1,2,3,4,5,6,7,8,9,10,11,12,13]):
    dataset = CIKMCUPDataset(config.root_path)

    data_dict = {}
    for client_id in range(1,len(dataset)+1):
        if client_id not in used_id:
          continue
        dataloader_dict = {}
        dataloader_dict['train'] = DataLoader(dataset[client_id]['train'],config.batch_size,shuffle=config.shuffle)
        dataloader_dict['val'] = DataLoader(dataset[client_id]['val'],config.batch_size,shuffle=False)
        dataloader_dict['test'] = DataLoader(dataset[client_id]['test'],config.batch_size,shuffle=False)
        data_dict[client_id] = dataloader_dict
    return data_dict


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# 支持多分类和二分类
class FocalLoss(nn.Module):
  """
  This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
  'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
  :param num_class:
  :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
  :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
          focus on hard misclassified example
  :param smooth: (float,double) smooth value when cross entropy
  :param balance_index: (int) balance class index, should be specific when alpha is float
  :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
  """
 
  def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
    super(FocalLoss, self).__init__()
    self.num_class = num_class
    self.alpha = alpha
    self.gamma = gamma
    self.smooth = smooth
    self.size_average = size_average
 
    if self.alpha is None:
      self.alpha = torch.ones(self.num_class, 1)
    elif isinstance(self.alpha, (list, np.ndarray)):
      assert len(self.alpha) == self.num_class
      self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
      self.alpha = self.alpha / self.alpha.sum()
    elif isinstance(self.alpha, float):
      alpha = torch.ones(self.num_class, 1)
      alpha = alpha * (1 - self.alpha)
      alpha[balance_index] = self.alpha
      self.alpha = alpha
    else:
      raise TypeError('Not support alpha type')
 
    if self.smooth is not None:
      if self.smooth < 0 or self.smooth > 1.0:
        raise ValueError('smooth value should be in [0,1]')
 
  def forward(self, input, target):
    logit = F.softmax(input, dim=1)
 
    if logit.dim() > 2:
      # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
      logit = logit.view(logit.size(0), logit.size(1), -1)
      logit = logit.permute(0, 2, 1).contiguous()
      logit = logit.view(-1, logit.size(-1))
    target = target.view(-1, 1)
 
    # N = input.size(0)
    # alpha = torch.ones(N, self.num_class)
    # alpha = alpha * (1 - self.alpha)
    # alpha = alpha.scatter_(1, target.long(), self.alpha)
    epsilon = 1e-10
    alpha = self.alpha
    if alpha.device != input.device:
      alpha = alpha.to(input.device)
 
    idx = target.cpu().long()
    one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
    one_hot_key = one_hot_key.scatter_(1, idx, 1)
    if one_hot_key.device != logit.device:
      one_hot_key = one_hot_key.to(logit.device)
 
    if self.smooth:
      one_hot_key = torch.clamp(
        one_hot_key, self.smooth, 1.0 - self.smooth)
    pt = (one_hot_key * logit).sum(1) + epsilon
    logpt = pt.log()
 
    gamma = self.gamma
 
    alpha = alpha[idx]
    loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
    if self.size_average:
      loss = loss.mean()
    else:
      loss = loss.sum()
    return loss

# 可直接调用此函数
def set_seed(SEED=0):
  np.random.seed(SEED)
  torch.manual_seed(SEED)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
  torch.cuda.manual_seed(SEED)  # 为GPU设置随机种子
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
  torch.backends.cudnn.deterministic = True


def socre(base_error,val_error):
    s = 0
    client = 1
    for i,j in zip(base_error,val_error):
        s = (i-j)/i + s
        # print(j,'---',(i-j)/i)
        print('Client:{} Best Error:{:.7f} Score:{:.7f}'.format(client,j,(i-j)/i))
        client = client + 1
    return s/len(val_error)

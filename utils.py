import logging

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

def load_client_data(config):
    dataset = CIKMCUPDataset(config.root_path)

    data_dict = {}
    for client_id in range(1,len(dataset)+1):
        dataloader_dict = {}
        dataloader_dict['train'] = DataLoader(dataset[client_id]['train'],config.batch_size,shuffle=config.shuffle)
        dataloader_dict['val'] = DataLoader(dataset[client_id]['val'],config.batch_size,shuffle=False)
        dataloader_dict['test'] = DataLoader(dataset[client_id]['test'],config.batch_size)
        data_dict[client_id] = dataloader_dict
    return data_dict
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, \
    global_max_pool

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net, GINE_Net
from federatedscope.gfl.model.gpr import GPR_Net

EMD_DIM = 200


class AtomEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i in range(in_channels):
            emb = torch.nn.Embedding(EMD_DIM, hidden)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class GNN_C(torch.nn.Module):
    r"""GNN model with pre-linear layer, pooling layer
        and output layer for graph classification tasks.

    Arguments:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden (int): hidden dim for all modules.
        max_depth (int): number of layers for gnn.
        dropout (float): dropout probability.
        gnn (str): name of gnn type, use ("gcn" or "gin").
        pooling (str): pooling method, use ("add", "mean" or "max").
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 pooling='add',
                 use_edge=False,
                 use_edge_c=False,
                 edge_dim=10,
                 use_eps=False):
        super(GNN_C, self).__init__()
        self.use_edge=use_edge
        self.use_edge_c=use_edge_c 
        self.dropout = dropout
        # Embedding (pre) layer
        # self.encoder_atom = AtomEncoder(in_channels, hidden)
        self.encoder = Linear(in_channels, hidden)
        # self.encoder = Sequential(
        #                 Linear(in_channels,hidden),
        #                 torch.nn.BatchNorm1d(hidden)
        # )
        # GNN layer
        self.gnn = GIN_Net(in_channels=hidden,
                            out_channels=hidden,
                            hidden=hidden,
                            max_depth=max_depth,
                            dropout=dropout)
        if use_edge==True:
            self.gnn = GINE_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout,
                                edge_dim=edge_dim,
                                use_eps=use_eps)

        self.gnn_c = GIN_Net(in_channels=hidden,
                            out_channels=hidden,
                            hidden=hidden,
                            max_depth=max_depth,
                            dropout=dropout)
        
        if use_edge_c==True:
            self.gnn_c = GINE_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout,
                                edge_dim=edge_dim,
                                use_eps=use_eps)

        # Pooling layer
        if pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(f'Unsupported pooling type: {pooling}.')
        # Output layer
        self.linear = Sequential(Linear(2*hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        if x.dtype == torch.int64:
            print('atom')
            x = self.encoder_atom(x)
        else:
            x = self.encoder(x)

        # x = self.encoder(x)
        if self.use_edge== True:
            edge_attr = data.edge_attr
            x = self.gnn((x,edge_index,edge_attr))
            if self.use_edge_c == True:
                x_c = self.gnn_c((x,edge_index,edge_attr))
            else:
                x_c = self.gnn_c((x,edge_index))
        else:
            x = self.gnn((x, edge_index))
            if self.use_edge_c == True:
                x_c = self.gnn_c((x,edge_index,edge_attr))
            else:
                x_c = self.gnn_c((x,edge_index))
        x = self.pooling(x, batch)
        x_c = self.pooling(x_c, batch)
        x = torch.cat([x,x_c],dim=1)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x

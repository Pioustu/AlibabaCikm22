import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn.modules.batchnorm import BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv,GraphConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.nn import BatchNorm1d, Identity

class GCN(torch.nn.Module):
    def __init__(self,
                in_dim,
                in_channels,
                out_channels,
                num_cls,
                max_depth=2,
                hidden=64,
                drop_out=0.2
                ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels=in_dim, out_channels=hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden, num_cls)

    def forward(self,data):
        x, edge_index, batch = data.x,data.edge_index,data.batch
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

class GNN(torch.nn.Module):
    def __init__(self,
                in_dim,
                in_channels,
                out_channels,
                num_cls,
                max_depth=2,
                hidden=64,
                drop_out=0.2
                ):
        super(GNN,self).__init__()
        torch.manual_seed(12345)
        self.stage1 = Linear(in_dim,in_channels)
        self.conv1 = GraphConv(in_channels=in_channels,out_channels=hidden)
        self.conv2 = GraphConv(in_channels=hidden, out_channels=2*hidden)
        self.conv3 = GraphConv(in_channels=2*hidden, out_channels=out_channels)
        self.lin = Linear(out_channels, num_cls)
    
    def forward(self, data):
        x = self.stage1(data.x)
        data.x = x
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x,edge_index))
        x = F.relu(self.conv2(x,edge_index))
        x = F.relu(self.conv3(x,edge_index))
        x = global_mean_pool(x,data.batch)
        # x = F.softmax(self.lin)
        x = self.lin(F.dropout(x))
        return x
# class GNN(torch.nn.Module):
#     def __init__(self,
#                 in_dim,
#                 in_channels,
#                 out_channels,
#                 num_cls,
#                 max_depth=2,
#                 hidden=64,
#                 drop_out=0.2
#                 ):
#         super(GNN,self).__init__()
#         in_channels=in_dim
#         self.conv1 = GraphConv(in_channels=in_channels,out_channels=hidden)
#         self.conv2 = GraphConv(in_channels=hidden, out_channels=2*hidden)
#         self.conv3 = GraphConv(in_channels=2*hidden, out_channels=out_channels)
#         self.lin = Linear(out_channels, num_cls)
    
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = F.relu(self.conv1(x,edge_index))
#         x = F.relu(self.conv2(x,edge_index))
#         x = F.relu(self.conv3(x,edge_index))
#         x = global_mean_pool(x,data.batch)
#         # x = F.softmax(self.lin)
#         x = self.lin(x)
#         return x


class GCN_C(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 out_channels,
                 num_cls,
                 max_depth=2,
                 hidden=64,
                 drop_out=0.2
                 ):
        super(GCN_C, self).__init__()
        self.stage1 = nn.Sequential(
                        nn.Linear(in_features=in_dim,out_features=in_channels)
                        # nn.LayerNorm(in_channels)
        )
        self.feature = GCN_Net(in_channels=in_channels,out_channels=out_channels,hidden=hidden,max_depth=max_depth)
        self.classfer = nn.Sequential(
                        nn.Dropout(drop_out),
                        # nn.LayerNorm(out_channels),
                        nn.Linear(out_channels,num_cls),
                        nn.Softmax()
        )
    
    def forward(self,data):
        x = self.stage1(data.x)
        data.x = x
        x = self.feature(data)
        x = global_mean_pool(x,data.batch)
        x = self.classfer(x)
        return x

class GCN_R(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 out_channels,
                 num_cls,
                 max_depth=2,
                 hidden=64,
                 drop_out=0.2
                 ):
        super(GCN_R, self).__init__()
        self.stage1 = nn.Sequential(
                        nn.Linear(in_features=in_dim,out_features=in_channels)
                        # nn.LayerNorm(in_channels)
        )
        self.feature = GCN_Net(in_channels=in_channels,out_channels=out_channels,hidden=hidden,max_depth=max_depth)
        self.classfer = nn.Sequential(
                        # nn.LayerNorm(out_channels),
                        nn.Dropout(drop_out),
                        nn.Linear(out_channels,num_cls)

        )
    
    def forward(self,data):
        x = self.stage1(data.x)
        data.x=x
        x = self.feature(data)
        x = global_mean_pool(x,batch=data.batch)
        x = self.classfer(x)
        return x

class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GCN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout
            

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GIN_R(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 out_channels,
                 num_cls,
                 max_depth=2,
                 hidden=64,
                 drop_out=0.2
                 ):
        super(GIN_R, self).__init__()
        self.stage1 = nn.Sequential(
                        nn.Linear(in_features=in_dim,out_features=in_channels)
                        # nn.LayerNorm(in_channels)
        )
        in_channels = in_dim
        self.feature = GIN_Net(in_channels=in_channels,out_channels=out_channels,hidden=hidden,max_depth=max_depth)
        self.classfer = nn.Sequential(
                        # nn.LayerNorm(out_channels),
                        nn.Dropout(drop_out),
                        nn.Linear(out_channels,num_cls)

        )
    
    def forward(self,data):
        # x = self.stage1(data.x)
        # data.x=x
        x = self.feature(data)
        x = global_mean_pool(x,batch=data.batch)
        x = self.classfer(x)
        return x

class GIN_C(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 in_channels,
                 out_channels,
                 num_cls,
                 max_depth=2,
                 hidden=64,
                 drop_out=0.2
                 ):
        super(GIN_C, self).__init__()
        self.stage1 = nn.Sequential(
                        nn.Linear(in_features=in_dim,out_features=in_channels)
                        # nn.LayerNorm(in_channels)
        )
        in_channels = in_dim
        self.feature = GIN_Net(in_channels=in_channels,out_channels=out_channels,hidden=hidden,max_depth=max_depth)
        self.classfer = nn.Sequential(
                        # nn.LayerNorm(out_channels),
                        nn.Dropout(drop_out),
                        nn.Linear(out_channels,num_cls),
                        nn.Softmax()

        )
    
    def forward(self,data):
        # x = self.stage1(data.x)
        # data.x=x
        x = self.feature(data)
        x = global_mean_pool(x,batch=data.batch)
        x = self.classfer(x)
        return x


class GIN_Net(torch.nn.Module):
    r"""Graph Isomorphism Network model from the "How Powerful are Graph
    Neural Networks?" paper, in ICLR'19

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GIN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINConv(MLP([in_channels, hidden, hidden],
                                batch_norm=True)))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINConv(
                        MLP([hidden, hidden, out_channels], batch_norm=True)))
            else:
                self.convs.append(
                    GINConv(MLP([hidden, hidden, hidden], batch_norm=True)))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self,
                 channel_list,
                 dropout=0.,
                 batch_norm=True,
                 relu_first=False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.linears = ModuleList()
        self.norms = ModuleList()
        for in_channel, out_channel in zip(channel_list[:-1],
                                           channel_list[1:]):
            self.linears.append(Linear(in_channel, out_channel))
            self.norms.append(
                BatchNorm1d(out_channel) if batch_norm else Identity())

    def forward(self, x):
        x = self.linears[0](x)
        for layer, norm in zip(self.linears[1:], self.norms[:-1]):
            if self.relu_first:
                x = F.relu(x)
            x = norm(x)
            if not self.relu_first:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer.forward(x)
        return x
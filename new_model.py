from turtle import forward
import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn.modules.container import Sequential
from torch_geometric.nn.glob import GraphMultisetTransformer
from torch_geometric.nn import GINConv,GINEConv
from torch.nn import ModuleList, BatchNorm1d, Identity
from torch_geometric.data import Data

from torch_geometric.nn.conv import GCNConv, GraphConv,GATConv
from torch_geometric.nn.glob.glob import global_mean_pool
class GTR(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 max_depth=2,
                 dropout=.0,
                 use_edge=False,
                 edge_dim=10):
        super(GTR, self).__init__()

        # 默认参数
        self.use_edge=use_edge
        # self.ln1 = torch.nn.LayerNorm(hidden)
        # self.ln2 = torch.nn.LayerNorm(hidden)
        self.encoder = Linear(in_channels, hidden) # Encoder 用于将节点特征进行编码
        if use_edge == True:
            self.gnn = GINE_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout,edge_dim=edge_dim)
            # self.gnn = GAT_Net(in_channels=hidden,out_channels=hidden,heads=head,hidden=hidden,max_depth=max_depth,dropout=dropout,edge_dim=edge_dim)
        else:
            self.gnn = GIN_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout)
        # gin = GIN_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout)
        self.gtr = GraphMultisetTransformer(in_channels=hidden,hidden_channels=hidden,out_channels=hidden,Conv=GATConv,layer_norm=True)
        self.clf = Linear(hidden,out_channels)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x) # 编码
        # x = self.ln1(x)
        if self.use_edge == True: # gnn提取特征
            edge_attr = data.edge_attr 
            x = self.gnn((x,edge_index,edge_attr))
        else:
            x = self.gnn((x,edge_index))
        x = self.gtr(x,batch,edge_index) # 图表示 
        # x = self.ln2(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.clf(x) # 分类/回归
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
        x = self.norms[-1](x) # 个人修改
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
                 dropout=.0,
                 use_res=False,
                 batch_norm=True):
        super(GIN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINConv(MLP([in_channels, hidden, hidden],
                                batch_norm=batch_norm)))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINConv(
                        MLP([hidden, hidden, out_channels], batch_norm=batch_norm)))
            else:
                self.convs.append(
                    GINConv(MLP([hidden, hidden, hidden], batch_norm=batch_norm)))
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

class GINE_Net(torch.nn.Module):
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
                 dropout=.0,
                 edge_dim=None,
                 use_eps=False,
                 use_res=False):
        super(GINE_Net, self).__init__()
        self.convs = ModuleList()
        self.use_res=use_res
        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINEConv(MLP([in_channels, hidden, hidden],
                                batch_norm=True),
                                train_eps=use_eps,
                                edge_dim=edge_dim))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINEConv(
                        MLP([hidden, hidden, out_channels], batch_norm=True), 
                                train_eps=use_eps,
                                edge_dim=edge_dim))
            else:
                self.convs.append(
                    GINEConv(MLP([hidden, hidden, hidden], batch_norm=True),
                                train_eps=use_eps,
                                edge_dim=edge_dim))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        elif isinstance(data, tuple):
            x, edge_index, edge_attr = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x
        
if __name__ == '__main__':
    my_gtr = GTR(20,2,128,use_edge=True,edge_dim=10)
    print(my_gtr)
    fed_layer = []
    for name,value in my_gtr.named_parameters():
        if 'norm' not in name and '.lin.' not in name:
            fed_layer.append(name)
    print(fed_layer)
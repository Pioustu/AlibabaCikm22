from turtle import forward
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool
from torch.nn import ModuleList
from torch.nn import BatchNorm1d, Identity
from torch_geometric.nn import GINConv,GINEConv

class Alpha(torch.nn.Module):
    def __init__(self):
        super(Alpha,self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(13))
    def forward(self,data):
        alpha = torch.softmax(self.alpha,dim=0)
        tmp = sum([alpha[i]*data[i] for i in range(len(alpha))])
        return tmp

        


class GNN3(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden,
                 max_depth=2,
                 dropout=.0,
                 use_edge=False,
                 p_gnn=False,
                 edge_dim=10):
        super(GNN3, self).__init__()
        self.use_pgnn = p_gnn
        self.use_edge=use_edge
        self.dropout = dropout

        self.encoder = Linear(in_channels, hidden)
        self.gbn = BatchNorm1d(hidden)
        self.ggnn = GIN_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout)
        self.pooling = global_mean_pool
        # self.glinear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.linear = Sequential(Linear(hidden, hidden), torch.nn.GELU())

        if p_gnn == True:
            if use_edge == True:
                self.pgnn = GINE_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout,edge_dim=edge_dim)
            else:
                self.pgnn = GIN_Net(in_channels=hidden,out_channels=hidden,hidden=hidden,max_depth=max_depth,dropout=dropout)
            # self.plinear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
            self.linear = Sequential(Linear(2*hidden, hidden), torch.nn.GELU())
            self.clf = Linear(hidden,out_channels)
        else:
            self.clf = Linear(hidden,out_channels)
    
    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        x_g = self.gbn(x)
        x_g = self.ggnn((x, edge_index))
        x_g = self.pooling(x_g, batch)
        # x_g = self.glinear(x_g)
        if self.use_pgnn == True: # 开启私有分支
            if self.use_edge == True:
                edge_attr = data.edge_attr
                x_p = self.pgnn((x,edge_index,edge_attr))
            else:
                x_p = self.pgnn((x,edge_index))
            x_p = self.pooling(x_p,batch)
            # x_p = self.plinear(x_p)
            x = torch.cat([x_g,x_p],dim=1)
            # x = x_g + x_p
        else:
            x = x_g
        x = self.linear(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.clf(x)
        return x





class GNN2(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 pooling='add',
                 use_edge=False,
                 onet = False,
                 use_edge_c=False,
                 edge_dim=10,
                 use_eps=False,
                 use_res=False):
        super(GNN2, self).__init__()
        self.use_edge=use_edge
        self.use_edge_c=use_edge_c 
        self.dropout = dropout
        self.onet=onet
        self.use_res=use_res

        self.encoder = Linear(in_channels, hidden)

        self.gnn = GIN_Net(in_channels=hidden,
                            out_channels=hidden,
                            hidden=hidden,
                            max_depth=max_depth,
                            dropout=dropout,
                            use_res=use_res)
        if use_edge==True:
            self.gnn = GINE_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout,
                                edge_dim=edge_dim,
                                use_eps=use_eps,
                                use_res=use_res)
        if onet==True: # 双网络结构
            self.BN = BatchNorm1d(hidden)
            self.gnn_c = GIN_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)
            self.alpha = torch.nn.Parameter(torch.ones(2))
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
        if onet==True:
            self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
            self.linear_c = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
            self.clf = Linear(2*hidden,out_channels)
        else:
            self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
            self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.encoder(x)

        # x = self.encoder(x)
        if self.use_edge== True:
            edge_attr = data.edge_attr
            if self.use_res==True:# 使用残差
                x = x + self.gnn((x,edge_index,edge_attr))
            else:
                x = self.gnn((x,edge_index,edge_attr))
            if self.onet==True:
                x = self.BN(x) # 使用全局bn
                if self.use_edge_c == True:
                    x_c = self.gnn_c((x,edge_index,edge_attr))
                else:
                    x_c = self.gnn_c((x,edge_index))
        else:
            if self.use_res==True:# 使用残差
                x = x + self.gnn((x, edge_index))
            else:
                x = self.gnn((x, edge_index))
            if self.onet==True:
                x = self.BN(x)
                if self.use_edge_c == True:
                    x_c = self.gnn_c((x,edge_index,edge_attr))
                else:
                    x_c = self.gnn_c((x,edge_index))
        x = self.pooling(x, batch)

        x = self.linear(x)

        if self.onet==True: # 双路时候使用
            x_c = self.pooling(x_c, batch)
            x_c = self.linear_c(x_c)
            x = torch.cat([x,x_c],dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 pooling='add',
                 use_edge=False,
                 onet = False,
                 use_edge_c=False,
                 edge_dim=10,
                 use_eps=False,
                 use_res=False):
        super(GNN, self).__init__()
        self.use_edge=use_edge
        self.use_edge_c=use_edge_c 
        self.dropout = dropout
        self.onet=onet
        self.use_res=use_res

        self.encoder = Linear(in_channels, hidden)

        self.gnn = GIN_Net(in_channels=hidden,
                            out_channels=hidden,
                            hidden=hidden,
                            max_depth=max_depth,
                            dropout=dropout,
                            use_res=use_res)
        if use_edge==True:
            self.gnn = GINE_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout,
                                edge_dim=edge_dim,
                                use_eps=use_eps,
                                use_res=use_res)
        if onet==True: # 双网络结构
            self.gnn_c = GIN_Net(in_channels=hidden,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)
            self.alpha = torch.nn.Parameter(torch.ones(2))
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
        if onet==True:
            self.linear = Sequential(Linear(2*hidden, hidden), torch.nn.ReLU())
        else:
            self.linear = Sequential(Linear(hidden, hidden), torch.nn.ReLU())
        self.clf = Linear(hidden, out_channels)

    def forward(self, data):
        if isinstance(data, Batch):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        elif isinstance(data, tuple):
            x, edge_index, batch = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.encoder(x)

        # x = self.encoder(x)
        if self.use_edge== True:
            edge_attr = data.edge_attr
            if self.use_res==True:# 使用残差
                x = x + self.gnn((x,edge_index,edge_attr))
            else:
                x = self.gnn((x,edge_index,edge_attr))
            if self.onet==True:
                if self.use_edge_c == True:
                    x_c = self.gnn_c((x,edge_index,edge_attr))
                else:
                    x_c = self.gnn_c((x,edge_index))
        else:
            if self.use_res==True:# 使用残差
                x = x + self.gnn((x, edge_index))
            else:
                x = self.gnn((x, edge_index))
            if self.onet==True:
                if self.use_edge_c == True:
                    x_c = self.gnn_c((x,edge_index,edge_attr))
                else:
                    x_c = self.gnn_c((x,edge_index))
        x = self.pooling(x, batch)
        if self.onet==True:
            x_c = self.pooling(x_c, batch)
            # alpha = torch.softmax(self.alpha,dim=0)
            # x = alpha[0]*x + alpha[1]*x_c
            x = torch.cat([x,x_c],dim=1)
        x = self.linear(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.clf(x)
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
        self.use_res=use_res
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
            if self.use_res==True:
                x = x + conv(x, edge_index)
            else:
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
            if self.use_res==True:
                x = x + conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index, edge_attr)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x





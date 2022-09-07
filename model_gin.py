import torch_scatter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GINConv
from torch.nn import Parameter, Module, Sigmoid
import torch
from torch.nn import Linear,ReLU
from torch.nn.modules.container import Sequential
from torch_geometric.nn.glob import global_add_pool

class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()
        
        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)

class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps
    
    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res

class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=32, **kwargs):
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units
    
    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        x = x.view((-1, self.dim * self.units))
        x = self.mlp(x)
        return x

class GINPNAConv(GINConv):
    def __init__(self, nn, node_dim=32, **kwargs):
        super(GINPNAConv, self).__init__(nn, **kwargs)
        self.mlp = torch.nn.Linear(node_dim*12, node_dim)
        self.delta = 2.5749
    
    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
        
        aggrs = [sums, maxs, means, var]
        c_idx = index.bincount().float().view(-1, 1)
        l_idx = torch.log(c_idx + 1.)
        
        amplification_scaler = [c_idx / self.delta * a for a in aggrs]
        attenuation_scaler = [self.delta / c_idx * a for a in aggrs]
        combinations = torch.cat(aggrs+ amplification_scaler+ attenuation_scaler, dim=1)
        x = self.mlp(combinations)
    
        return x

class LAFNet(torch.nn.Module):
    def __init__(self,in_channel,out_channel,node_dim):
        super(LAFNet, self).__init__()
        
        self.encoder = Linear(node_dim,in_channel)

        nn1 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv1 = GINLAFConv(nn1, units=3, node_dim=in_channel)
        self.bn1 = torch.nn.BatchNorm1d(in_channel)

        nn2 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv2 = GINLAFConv(nn2, units=3, node_dim=in_channel)
        self.bn2 = torch.nn.BatchNorm1d(in_channel)

        nn3 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv3 = GINLAFConv(nn3, units=3, node_dim=in_channel)
        self.bn3 = torch.nn.BatchNorm1d(in_channel)

        nn4 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv4 = GINLAFConv(nn4, units=3, node_dim=in_channel)
        self.bn4 = torch.nn.BatchNorm1d(in_channel)

        nn5 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv5 = GINLAFConv(nn5, units=3, node_dim=in_channel)
        self.bn5 = torch.nn.BatchNorm1d(in_channel)

        self.fc1 = Linear(in_channel, in_channel)
        self.fc2 = Linear(in_channel, out_channel)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x

class PNANet(torch.nn.Module):
    def __init__(self,in_channel,out_channel,node_dim):
        super(PNANet, self).__init__()

        self.encoder = Linear(node_dim,in_channel)

        nn1 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv1 = GINPNAConv(nn1, node_dim=in_channel)
        self.bn1 = torch.nn.BatchNorm1d(in_channel)

        nn2 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv2 = GINPNAConv(nn2, node_dim=in_channel)
        self.bn2 = torch.nn.BatchNorm1d(in_channel)

        nn3 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv3 = GINPNAConv(nn3, node_dim=in_channel)
        self.bn3 = torch.nn.BatchNorm1d(in_channel)

        nn4 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv4 = GINPNAConv(nn4, node_dim=in_channel)
        self.bn4 = torch.nn.BatchNorm1d(in_channel)

        nn5 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv5 = GINPNAConv(nn5, node_dim=in_channel)
        self.bn5 = torch.nn.BatchNorm1d(in_channel)

        self.fc1 = Linear(in_channel, in_channel)
        self.fc2 = Linear(in_channel, out_channel)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F

class GINNet(torch.nn.Module):
    def __init__(self,in_channel,out_channel,node_dim):
        super(GINNet, self).__init__()
        self.encoder = Linear(node_dim,in_channel)

        nn1 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(in_channel)

        nn2 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(in_channel)

        nn3 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(in_channel)

        nn4 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(in_channel)

        nn5 = Sequential(Linear(in_channel, in_channel), ReLU(), Linear(in_channel, in_channel))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(in_channel)

        self.fc1 = Linear(in_channel, in_channel)
        self.fc2 = Linear(in_channel, out_channel)

    def forward(self,data):
        x, edge_index, batch = data.x,data.edge_index,data.batch
        x = self.encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
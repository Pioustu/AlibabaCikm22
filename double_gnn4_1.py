import enum
import copy
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.functional import batch_norm
from torch.nn.modules.batchnorm import BatchNorm1d

from model import GIN_Net,GNN
from yacs.config import CfgNode
from tensorboardX import SummaryWriter

from client import Client

from utils import load_client_data,set_seed,socre
from server import FedBNGnn2

# 由官方给给定
base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]

# 用于计算blance weight，通过client train dataset比例来确定
alpha_all_ = [
            [0.611689351481185, 0.388310648518815],
            [0.5911602209944752, 0.4088397790055249],
            [0.5561063542136098, 0.4438936457863903],
            [0.3465346534653465, 0.6534653465346535],
            [0.6117021276595744, 0.3882978723404255],
            [0.2670299727520436, 0.7329700272479565],
            [0.5547576301615799, 0.4452423698384201],
            [0.16602316602316602, 0.833976833976834]
]
alpha_all = [[a[1],a[0]] for a in alpha_all_ ]

tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/doublegann4_1/score'
config_path = './config/double_gnn2.yaml'
client_model_save_path = '/home/featurize/cikm22/result/doublegann4_1/'
client_tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/doublegann4_1/'

if __name__ == '__main__':
    set_seed()
    
    rr = 0

    global_gnn = GIN_Net(in_channels=128,
                                out_channels=128,
                                hidden=128,
                                max_depth=3,
                                dropout=0.3,
                                batch_norm=True).cuda()

    global_bn = BatchNorm1d(128).cuda()
    # tensorboard保存地址
    writer = SummaryWriter(tensorboard_path)

    # 加载配置文件
    config = CfgNode.load_cfg(open(config_path))

    # 加载所有的数据
    all_dl = load_client_data(config.data_sp)

    # 保存所有的客户端
    all_client=[]

    # 客户端初始化
    for i in range(1,14):
        # if i in [1,2,3,4,5,6,7,8]:
        #     continue
        client_id = i
        client_name = 'client' + str(i)
        client_cfg = config[client_name]
        client_cfg.save_path = client_model_save_path
        client_cfg.writer_path =client_tensorboard_path
        client_train_dl = all_dl[i]['train']
        client_val_dl = all_dl[i]['val']
        client_test_dl = all_dl[i]['test']
        # client_cfg.is_ep = 1
        # 对分类的进行添加alpha
        if i<=8:
            alpha = alpha_all[i-1]
        else:
            alpha = None
        focal=False
        client_cfg.is_lr = 0.001
        client_cfg.ci_lr = 0.001
        client_cfg.ft_lr = 0.001
        client_cfg.is_ep = 1
        client_cfg.ft_ep = 5
        # if i in [2,5,11,12]: # 2，5，11，12 pgnn不使用边
        #     client_cfg.edge_dim=0
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,double=True,use_teacher=False,use_dg=True,use_pgnn=True)
        # client.model.ggnn = global_gnn # 共用一个gnn
        client.model.gbn = global_bn # 共用一个BN
        # 初始化相同的参数
        tmp ={}
        for name,value in client.model.pgnn.state_dict().items():
            if name in global_gnn.state_dict().keys():
                tmp[name] = global_gnn.state_dict()[name]
            else:
                tmp[name] = value
        client.model.pgnn.load_state_dict(tmp)

        all_client.append(client)
    
    ROUND = 500

    global_gnn_opt = torch.optim.Adamax(global_gnn.parameters(),lr=0.05)
    global_bn_opt = torch.optim.Adamax(global_bn.parameters(),lr=0.05)

    for round in range(ROUND):

        # 单独训练一轮
        for j, client in enumerate(all_client):
            client.update(encoder=True,gnn=False,ggnn=True,pgnn=True,linear=True,glinear=False,plinear=False,clf=True,stage='is',kd=False)

        # 更新全局bn
        global_bn_opt.step()
        global_bn_opt.zero_grad()

        for j,client in enumerate(all_client):
            p_client = all_client[j-1]
            p_weight = copy.deepcopy(p_client.model.state_dict())
            client.load_mate(p_weight,keep_pgnn=True)

        # gnn 和 gnnc训练
        for j, client in enumerate(all_client):
            client.update(encoder=False,gnn=False,ggnn=True,pgnn=False,linear=False,glinear=False,plinear=False,clf=False,stage='ci',kd=False)

        # 更新全局gnn
        global_bn_opt.step()
        global_bn_opt.zero_grad()

        
        # 其他层训练
        val_error = []
        best_val_error = []
        for i, client in enumerate(all_client):
            client.update(encoder=True,gnn=False,ggnn=False,pgnn=False,linear=True,glinear=False,plinear=False,clf=True,stage='ft',kd=False)
            val_error.append(client.error)
            best_val_error.append(client.best_error)
        
        # global_gnn_opt.zero_grad()
        global_bn_opt.zero_grad()

        # 计算得分
        res = socre(base_error,val_error)
        best_res = socre(base_error,best_val_error)
        print('Round:{} Best Score:{} Now Score:{} '.format(round,best_res,res))
        writer.add_scalar('Score', res, round)
        writer.add_scalar('Best Score', best_res, round)


'''
    使用circle训练策略,每轮包含一次gnn整体更新和一次自己更新
'''

import enum
import copy
from tokenize import group
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


from yacs.config import CfgNode
from tensorboardX import SummaryWriter

from client import Client
from utils import load_client_data,set_seed,socre
from server import FedBN, FedBN_GNN, FedBn, FedBnAp, FedCos, fedAvg
from model_gin import GINNet,PNANet,LAFNet

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

tensorboard_path = './cikm22/exp/tensorboard_logs/circle_gtr_clsusefull_prox_is3_ft5_2/score'
config_path = './config/config.yaml'
client_model_save_path = './cikm22/result/circle_gtr_clsusefull_prox_is3_ft5_2/'
client_tensorboard_path = './cikm22/exp/tensorboard_logs/circle_gtr_clsusefull_prox_is3_ft5_2/'

fedlayer = ['gnn.convs.0.nn.linears.0.weight', 'gnn.convs.0.nn.linears.0.bias', 'gnn.convs.0.nn.linears.1.weight', 
            'gnn.convs.0.nn.linears.1.bias', 'gnn.convs.1.nn.linears.0.weight', 'gnn.convs.1.nn.linears.0.bias', 
            'gnn.convs.1.nn.linears.1.weight', 'gnn.convs.1.nn.linears.1.bias', 'gtr.lin1.weight', 'gtr.lin1.bias', 
            'gtr.lin2.weight', 'gtr.lin2.bias', 'gtr.pools.0.mab.fc_q.weight', 'gtr.pools.0.mab.fc_q.bias', 
            'gtr.pools.0.mab.layer_k.att_src', 'gtr.pools.0.mab.layer_k.att_dst', 'gtr.poolts.0.mab.layer_k.bias', 
            'gtr.pools.0.mab.layer_k.lin_src.weight', 'gtr.pools.0.mab.layer_v.att_src', 'gtr.pools.0.mab.layer_v.att_dst', 
            'gtr.pools.0.mab.layer_v.bias', 'gtr.pools.0.mab.layer_v.lin_src.weight', 'gtr.pools.0.mab.fc_o.weight', 'gtr.pools.0.mab.fc_o.bias', 
            'gtr.pools.1.mab.fc_q.weight', 'gtr.pools.1.mab.fc_q.bias', 'gtr.pools.1.mab.layer_k.weight', 'gtr.pools.1.mab.layer_k.bias',
            'gtr.pools.1.mab.layer_v.weight', 'gtr.pools.1.mab.layer_v.bias', 'gtr.pools.1.mab.fc_o.weight', 'gtr.pools.1.mab.fc_o.bias', 
            'gtr.pools.2.mab.fc_q.weight', 'gtr.pools.2.mab.fc_q.bias', 'gtr.pools.2.mab.layer_k.weight', 'gtr.pools.2.mab.layer_k.bias', 
            'gtr.pools.2.mab.layer_v.weight', 'gtr.pools.2.mab.layer_v.bias', 'gtr.pools.2.mab.fc_o.weight', 'gtr.pools.2.mab.fc_o.bias'
            ]

if __name__ == '__main__':
    set_seed()
    
    rr = 0

    # tensorboard保存地址
    writer = SummaryWriter(tensorboard_path)


    # 加载配置文件
    config = CfgNode.load_cfg(open(config_path))
    # 加载所有的数据
    config.root_path = './cikm22/data/full/'
    all_dl = load_client_data(config.data_sp,[1,2,3,4,5,6,7,8,9,10,11,12,13])

    # 保存所有的客户端
    all_client=[]

    # norm  .lin.  ln  encoder  reg   

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
        client_cfg.is_lr = 0.001
        client_cfg.ci_lr = 0.001
        client_cfg.ft_lr = 0.001
        client_cfg.is_ep = 3
        client_cfg.ft_ep = 5
        client_cfg.ci_ep = 1
        # client_cfg.is_ep = 1
        # 对分类的进行添加alpha
        if i<=8:
            alpha = alpha_all[i-1]
        else:
            alpha = None
        focal=False
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,use_res=False,use_teacher=False,use_gtr=True)
        all_client.append(client)
    
    P=[i-1 for i in [1,2,3,5,7]]
    N=[i-1 for i in [4,6,8]]

    group1=[]
    group2=[]

    for j,client in enumerate(all_client):
        if j in P:
            group1.append(client)
        else:
            group2.append(client)

    # 初始化一样的参数
    for j, client in enumerate(all_client):
        p_weight = copy.deepcopy(all_client[j-1].model.state_dict())
        client.load_p_weight(p_weight,fedlayer)

    for i in range(400):
        val_error = []
        best_val_error = []

        # for j, client in enumerate(all_client):
        #     client.load_self_best_gnn()
        # cws = []
        # for j, client in enumerate(all_client):
        #     cws.append(client.model.state_dict())
        # new_cws = FedCos(cws,fedlayer)
        # # print(new_cws)
         
        # for j, client in enumerate(all_client):
        #     client.model.load_state_dict(new_cws[j])
        #     client.prox_weight=new_cws[j]
        
        for j, client in enumerate(all_client):
            client.prox_weight = client.model.state_dict()
            p_weight = all_client[j-1].model.state_dict()
            client.load_p_weight(p_weight,fedlayer)
            client.update(encoder=True,gnn=True,gtr=True,linear=False,clf=True,stage='is',kd=False,prox=True)
        
        for j, client in enumerate(all_client):
            client.prox_weight = client.model.state_dict()
            client.update(encoder=True,gnn=False,gtr=False,linear=False,clf=True,stage='ft',kd=False,prox=False)
        
        # for j, client in enumerate(group2):
        #     p_weight = group1[j-1].model.state_dict()
        #     client.load_p_weight(p_weight,fedlayer)
        #     client.prox_weight = client.model.state_dict()
        #     client.update(encoder=True,gnn=True,gtr=True,linear=False,clf=True,stage='is',kd=False,prox=True)
        # for j, client in enumerate(all_client):
        #     p_client = all_client[j-1]
        #     p_weight = copy.deepcopy(p_client.model.state_dict())
        #     client.load_gnn(p_weight)
        #     client.update(encoder=False,gnn=True,gtr=True,linear=False,clf=False,stage='ci',kd=False)
        
        # for j, client in enumerate(all_client):
        #     client.update(encoder=True,gnn=False,gtr=True,linear=False,clf=True,stage='ft',kd=False)
        #     val_error.append(client.error)
        #     best_val_error.append(client.best_error)


        # for j, client in enumerate(all_client):
        #     client.update(encoder=False,gnn=False,linear=True,clf=False,stage='is',kd=False)

        # for j, client in enumerate(all_client):
        #     client.update(encoder=False,gnn=False,linear=False,clf=True,stage='is',kd=False)
        
        for j, client in enumerate(all_client):
            # client.update(encoder=True,gnn=True,linear=True,clf=True,stage='ft',kd=False)
            val_error.append(client.error)
            best_val_error.append(client.best_error)

        res = socre(base_error,val_error)
        best_res = socre(base_error,best_val_error)
        print('Round:{} Circle Best Score:{} Now Score:{} '.format(i,best_res,res))
        writer.add_scalar('Score', res, i)
        writer.add_scalar('Best Score', best_res, i)
        
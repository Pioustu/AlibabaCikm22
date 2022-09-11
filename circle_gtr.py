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
from server import FedBN, FedBN_GNN, FedBn, FedBnAp, FedCos

# 由官方给给定
base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]

# 用于计算blance weight，通过client train dataset比例来确定, 用于focal loss计算
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

# 相关文件的路径
# tensorboard_path：整体的score分值的tensorboard路径（需要修改）
# config_path：配置文件（不需要修改）
# client_model_save_path：client中模型的保存路径（需要修改）
# client_tensorbard_path：client中tensorboard路径（需要修改）
tensorboard_path = './circle_gtr_clsusefull_prox_is3_ft5/score'
config_path = './config/config.yaml'
client_model_save_path = './result/model/circle_gtr_clsusefull_prox_is3_ft5/'
client_tensorboard_path = './result/exp/circle_gtr_clsusefull_prox_is3_ft5/'

# 需要参与fed交互的模型层名称，主要包括模型中的gnn层和pooling层
fedlayer = ['gnn.convs.0.nn.linears.0.weight', 'gnn.convs.0.nn.linears.0.bias', 'gnn.convs.0.nn.linears.1.weight', 
            'gnn.convs.0.nn.linears.1.bias', 'gnn.convs.1.nn.linears.0.weight', 'gnn.convs.1.nn.linears.0.bias', 
            'gnn.convs.1.nn.linears.1.weight', 'gnn.convs.1.nn.linears.1.bias', 'gtr.lin1.weight', 'gtr.lin1.bias', 
            'gtr.lin2.weight', 'gtr.lin2.bias', 'gtr.pools.0.mab.fc_q.weight', 'gtr.pools.0.mab.fc_q.bias', 
            'gtr.pools.0.mab.layer_k.bias', 
            'gtr.pools.0.mab.layer_k.lin_src.weight', 
            'gtr.pools.0.mab.layer_v.bias', 'gtr.pools.0.mab.layer_v.lin_src.weight', 'gtr.pools.0.mab.fc_o.weight', 'gtr.pools.0.mab.fc_o.bias', 
            'gtr.pools.1.mab.fc_q.weight', 'gtr.pools.1.mab.fc_q.bias', 'gtr.pools.1.mab.layer_k.weight', 'gtr.pools.1.mab.layer_k.bias',
            'gtr.pools.1.mab.layer_v.weight', 'gtr.pools.1.mab.layer_v.bias', 'gtr.pools.1.mab.fc_o.weight', 'gtr.pools.1.mab.fc_o.bias', 
            'gtr.pools.2.mab.fc_q.weight', 'gtr.pools.2.mab.fc_q.bias', 'gtr.pools.2.mab.layer_k.weight', 'gtr.pools.2.mab.layer_k.bias', 
            'gtr.pools.2.mab.layer_v.weight', 'gtr.pools.2.mab.layer_v.bias', 'gtr.pools.2.mab.fc_o.weight', 'gtr.pools.2.mab.fc_o.bias'
            ]

if __name__ == '__main__':
    # 设置随机种子
    set_seed()

    # 初始化tensorboard
    writer = SummaryWriter(tensorboard_path)

    # 加载配置文件
    config = CfgNode.load_cfg(open(config_path))

    # 加载所有的数据，使用full_data.py处理获取到的补齐的分类数据，地址为补齐后的数据地址
    all_dl = load_client_data(config.data_sp,[1,2,3,4,5,6,7,8,9,10,11,12,13])

    # 用于保存所有的客户端
    all_client=[]

    # 客户端初始化
    for i in range(1,14):
        # if i in [1,2,3,4,5,6,7,8]:
        #     continue
        # 加载数据
        client_id = i
        client_name = 'client' + str(i)
        client_cfg = config[client_name]
        client_cfg.save_path = client_model_save_path
        client_cfg.writer_path =client_tensorboard_path
        client_train_dl = all_dl[i]['train']
        client_val_dl = all_dl[i]['val']
        client_test_dl = all_dl[i]['test']

        # 设置不同阶段的学习率和更新轮次
        client_cfg.is_lr = 0.001
        client_cfg.ci_lr = 0.001
        client_cfg.ft_lr = 0.001
        client_cfg.is_ep = 3
        client_cfg.ft_ep = 5
        client_cfg.ci_ep = 1

        if i<=8:
            alpha = alpha_all[i-1]
        else:
            alpha = None
        focal=False

        # 初始化客户端
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,use_res=False,use_teacher=False,use_gtr=True)
        all_client.append(client)
    
    # 初始化一样的参数，将上一个客户端的参数load到当前客户端中
    for j, client in enumerate(all_client):
        p_weight = copy.deepcopy(all_client[j-1].model.state_dict())
        client.load_p_weight(p_weight,fedlayer) 

    # 设置更新400Round
    for i in range(370):

        # 记录每个客户端的得分
        val_error = []
        best_val_error = []

        # 第一个阶段：环形更新，每次第j个客户端会使用第j-1个模型参数进行更新，更新模型中的所有层（模型包含：encoder, gnn, gtr, clf），使用prox正则策略
        # 1、获取正则参数，使用第j个客户端模型参数作为正则参数
        # 2、获取j-1客户端的模型参数，并且加载到第j个客户端模型中
        # 3、使用第j个客户端中的模型更新整体模型
        for j, client in enumerate(all_client):
            client.prox_weight = client.model.state_dict()
            p_weight = all_client[j-1].model.state_dict()
            client.load_p_weight(p_weight,fedlayer)
            client.update(encoder=True,gnn=True,gtr=True,linear=False,clf=True,stage='is',kd=False,prox=True)
        
        # 第二个阶段：个性化调整，每个客户端独立更新私有层参数
        for j, client in enumerate(all_client):
            client.update(encoder=True,gnn=False,gtr=False,linear=False,clf=True,stage='ft',kd=False,prox=False)
             
        # 获取所有的client的分值
        for j, client in enumerate(all_client):
            val_error.append(client.error)
            best_val_error.append(client.best_error)

        res = socre(base_error,val_error)
        best_res = socre(base_error,best_val_error)
        print('Round:{} Circle Best Score:{} Now Score:{} '.format(i,best_res,res))
        writer.add_scalar('Score', res, i)
        writer.add_scalar('Best Score', best_res, i)
        
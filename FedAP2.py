import enum
import copy
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


from yacs.config import CfgNode
from tensorboardX import SummaryWriter

from client import Client
from utils import load_client_data,set_seed,socre
from server import FedAP, FedBN, FedBN_GNN, fedAvg

''' 使用逐层更新方式来进行FedAP训练 '''

# 由官方给给定
base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]

tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/FedAP2/score'
config_path = './config/config1.yaml'
client_model_save_path = '/home/featurize/cikm22/result/FedAP2/'
client_tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/FedAP2/'

if __name__ == '__main__':
    set_seed()
    
    rr = 0

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
        client_cfg.is_lr = 0.001
        client_cfg.ci_lr = 0.001
        client_cfg.ft_lr = 0.001
        client_cfg.is_ep = 3
        client_cfg.ci_ep = 1
        client_cfg.ft_ep = 3
        # client_cfg.is_ep = 1
        # 对分类的进行添加alpha
        focal=False
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True)
        all_client.append(client)
    

    ROUND = 1000
    for round in range(ROUND):
        val_error = []
        best_val_error = []

        # 训练所有层
        for i, client in enumerate(all_client):
            client.update(encoder=True,gnn=True,linear=True,clf=True,stage='is',kd=False)

        # 聚合使用FedAP
        cws = []
        for i, client in enumerate(all_client):
            cws.append(copy.deepcopy(client.model.state_dict()))
        # 聚合算法
        new_cws = FedAP(cws)
        # 聚合后结果返回给每个客户机
        for i, client in enumerate(all_client):
            client.model.load_state_dict(new_cws[i])
        
        # # 训练除了gnn其他层
        # for i, client in enumerate(all_client):
        #     client.update(encoder=True,gnn=False,linear=True,clf=True,stage='is',kd=False)
        
        # 微调其他层所有计算得分
        for i, client in enumerate(all_client):
            client.update(encoder=True,gnn=False,linear=True,clf=True,stage='ft',kd=False)
            val_error.append(client.error)
            best_val_error.append(client.best_error)

        # 计算得分
        res = socre(base_error,val_error)
        best_res = socre(base_error,best_val_error)
        print('Round:{} Best Score:{} Now Score:{} '.format(round,best_res,res))
        writer.add_scalar('Score', res, round)
        writer.add_scalar('Best Score', best_res, round)
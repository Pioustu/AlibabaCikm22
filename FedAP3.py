import enum
import copy
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


from yacs.config import CfgNode
from tensorboardX import SummaryWriter

from client import Client
from server import FedAP_Plus
from model import GNN
from utils import load_client_data,set_seed,socre
from server import FedAP, FedBN, FedBN_GNN, fedAvg

# 由官方给给定
base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]

tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/FedAP3/score'
config_path = './config/config1.yaml'
client_model_save_path = '/home/featurize/cikm22/result/FedAP3/'
client_tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/FedAP3/'

client_model_path = '/home/featurize/cikm22/result/circle/'

if __name__ == '__main__':
    set_seed()
    
    rr = 0

    start_gnn = GNN(in_channels=1,
                            out_channels=1,
                            hidden=128,
                            max_depth=3,
                            dropout=0.3,
                            pooling='mean',
                            use_edge=False).gnn
    start_linear = GNN(in_channels=1,
                            out_channels=1,
                            hidden=128,
                            max_depth=3,
                            dropout=0.3,
                            pooling='mean',
                            use_edge=False).linear

    # tensorboard保存地址
    writer = SummaryWriter(tensorboard_path)

    # 加载配置文件
    config = CfgNode.load_cfg(open(config_path))

    # 加载所有的数据
    all_dl = load_client_data(config.data_sp)

    # 保存所有的客户端
    all_client=[]

    # 客户端初始化
    for i in range(1,9):
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
        client_cfg.is_ep = 1
        client_cfg.ci_ep = 1
        client_cfg.ft_ep = 5
        # 对分类的进行添加alpha
        focal=False
        edge=True
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=edge)
        # 初始化gnn linear参数
        tmp ={}
        # for name,value in client.model.gnn.state_dict().items():
        #     if name in start_gnn.state_dict().keys():
        #         tmp[name] = start_gnn.state_dict()[name]
        #     else:
        #         tmp[name] = value
        # client.model.gnn.load_state_dict(tmp)
        # client.model.linear.load_state_dict(copy.deepcopy(start_linear.state_dict()))
        all_client.append(client)
    

        # 加载当前训练好的模型
        for i,client in enumerate(all_client):
            model_path = client_model_path + str(i+1) + '/best.pt'
            client.model.load_state_dict(torch.load(model_path))
            client.saveModel()
            client.best_error,_ = client.val(client.val_dl) # 从第一步后接着开始所以初始化best_error

    ROUND = 500
    for round in range(ROUND):

        # 训练
        for i, client in enumerate(all_client):
            client.update(encoder=True,gnn=True,linear=True,clf=True,stage='is',kd=False)

        # 收集所有的模型
        cws = []
        for i, client in enumerate(all_client):
            cws.append(copy.deepcopy(client.model.state_dict()))
        
        # 聚合算法
        new_cws = FedAP_Plus(cws)
        for i, client in enumerate(all_client):
            client.model.load_state_dict(new_cws[i])
        
        # 训练
        for i, client in enumerate(all_client):
            client.update(encoder=False,gnn=True,linear=False,clf=False,stage='is',kd=False,prox=True)
        
        # 微调
        val_error = []
        best_val_error = []
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
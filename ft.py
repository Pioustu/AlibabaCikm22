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
from server import FedBN, FedBN_GNN, fedAvg

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

config_path = './config/config1.yaml'
client_model_path = '/home/featurize/cikm22/result/circle/'
client_model_path_ft = '/home/featurize/cikm22/result/circle/ft/'
client_tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/ft/'

if __name__ == '__main__':
    set_seed()
    
    rr = 0

if __name__ == '__main__':
    set_seed()
    
    rr = 0

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
        client_cfg.save_path = client_model_path_ft
        client_cfg.writer_path =client_tensorboard_path
        client_train_dl = all_dl[i]['train']
        client_val_dl = all_dl[i]['val']
        client_test_dl = all_dl[i]['test']
        client_cfg.is_lr = 0.001
        client_cfg.ci_lr = 0.001
        client_cfg.ft_lr = 0.001
        client_cfg.is_ep = 1
        client_cfg.ft_ep = 300
        client_cfg.ci_ep = 1
        # client_cfg.is_ep = 1
        # 对分类的进行添加alpha
        if i<=8:
            alpha = alpha_all[i-1]
        else:
            alpha = None
        focal=False
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,use_res=False)
        all_client.append(client)
    
    # 加载微调前模型路径，将其保存到ft中，之后初始化best error之后进行微调
    for i,client in enumerate(all_client):
        model_path = client_model_path + str(i+1) + '/best.pt'
        client.model.load_state_dict(torch.load(model_path))
        client.t_model.load_state_dict(torch.load(model_path))
        client.cfg.save_path = client_model_path_ft
        client.saveModel()
        client.best_error,_ = client.val(client.val_dl) # 从第一步后接着开始所以初始化best_error

    # 开始微调
    val_error = []
    best_val_error = []
    for i,client in enumerate(all_client):
        client.update(stage='ft',encoder=True,gnn=False,linear=True,clf=True,kd=True)
        val_error.append(client.error)
        best_val_error.append(client.best_error)

    # for i,client in enumerate(all_client):
    #     client.update(stage='ft',encoder=True,gnn=True,linear=True,clf=True,kd=False)
    #     val_error.append(client.error)
    #     best_val_error.append(client.best_error)
    res = socre(base_error,val_error)
    best_res = socre(base_error,best_val_error)
    print('Circle Best Score:{} Now Score:{} '.format(best_res,res))

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

tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/iscrft/score'
config_path = './config/config1.yaml'
client_model_save_path_is = '/home/featurize/cikm22/result/is/'
client_model_save_path_ci = '/home/featurize/cikm22/result/is/ci/'
client_model_save_path_ft1 = '/home/featurize/cikm22/result/is/ci/ft1/'
client_model_save_path_ft2 = '/home/featurize/cikm22/result/is/ci/ft2/'
client_tensorboard_path = '/home/featurize/cikm22/exp/tensorboard_logs/iscrft2/'

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
        client_cfg.save_path = client_model_save_path_is
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
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,use_teacher=False)
        all_client.append(client)
    
    # isolated training
    # val_error = []
    # best_val_error = []
    # print('Start isolated training')
    # for i, client in enumerate(all_client):
    #     client.update(stage='is')
    #     best_val_error.append(client.best_error)
    #     val_error.append(client.error)
    
    # # print(best_val_error)

    # res = socre(base_error,val_error)
    # best_res = socre(base_error,best_val_error)
    # print('Isolated Best Score:{} Now Score:{} '.format(best_res,res))

    # writer.add_scalar('Score', res, rr)
    # writer.add_scalar('Best Score', best_res, rr)
    # rr = rr + 1


    # 读取is中最好的模型，然后将其保存到ci中
    for i,client in enumerate(all_client):
        model_path = client_model_save_path_is + str(i+1) + '/best.pt'
        client.model.load_state_dict(torch.load(model_path))
        client.cfg.save_path = client_model_save_path_ci
        client.saveModel()
        client.best_error,_ = client.val(client.val_dl) # 从第一步后接着开始所以初始化best_error

    # circle training
    print("Circle training")
    ci_no_update_stop = 150
    best = 0
    for cir in range(2000):
        if ci_no_update_stop<0 and cir>500:
            print('CI No Imporve SO Stop')
            break
        val_error = []
        best_val_error = []
        for i, client in enumerate(all_client):
            p_client = all_client[i-1]
            p_weight = copy.deepcopy(p_client.model.state_dict())
            client.load_gnn(p_weight)
            # client.load_linear(p_weight)
            client.update(encoder=False,gnn=True,linear=False,clf=False,stage='ci')
            best_val_error.append(client.best_error)
            val_error.append(client.error)
        res = socre(base_error,val_error)
        best_res = socre(base_error,best_val_error)
        print('Circle Best Score:{} Now Score:{} '.format(best_res,res))
        
        if best < best_res:
            if best == 0:
                ci_no_update_stop = 1000
            ci_no_update_stop = 150
            best = best_res

        ci_no_update_stop = ci_no_update_stop - 1

        writer.add_scalar('Score', res, rr)
        writer.add_scalar('Best Score', best_res, rr)
        rr = rr + 1

    # # ft1 固定gnn对其他层进行kd训练，首先保存所有最好模型到ft1中,并且加载最优模型
    # for i,client in enumerate(all_client):
    #     model_path = client_model_save_path_ci + str(i+1) + '/best.pt'
    #     client.model.load_state_dict(torch.load(model_path))
    #     client.cfg.save_path = client_model_save_path_ft1
    #     client.saveModel()

    '''单独微调设置'''
    # for i, client in enumerate(all_client):
    #     model_path = client_model_save_path_ft1 + str(i+1) + '/best.pt'
    #     client.model.load_state_dict(torch.load(model_path))
    #     client.best_error,_ = client.val(client.val_dl)
    #     client.cfg.save_path = client_model_save_path_ft1

    # # ft1
    val_error = []
    best_val_error = []
    for i, client in enumerate(all_client):
        client.update(stage='ft',encoder=True,gnn=False,linear=True,clf=True,kd=False)
        best_val_error.append(client.best_error)
        val_error.append(client.error)
    res = socre(base_error,val_error)
    best_res = socre(base_error,best_val_error)
    print('FT Best Score:{} Now Score:{} '.format(best_res,res))
    writer.add_scalar('Score', res, rr)
    writer.add_scalar('Best Score', best_res, rr)
    rr = rr + 1
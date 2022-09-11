import copy
from numpy import double
import torch
from re import I
from torch.nn.functional import batch_norm

from yacs.config import CfgNode

from client import Client
from utils import load_client_data,set_seed,socre
base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]


# 相关文件的路径
# model_file_path：模型保存的文件路径（需要修改）
# client_tensorboard_path：test时候tensorboard（需要修改）
# result_path：结果保存目录（需要修改）
# result_name：结果保存名称（需要修改）
model_file_path = './result/model/circle_gtr_clsusefull_prox_is3_ft5/'
# model_file_path_FedBn_clf = '/home/featurize/cikm22/result/only_cls_try_fed_ap/'
client_tensorboard_path = './cikm22/exp/tensorboard_logs/test/'
result_path = './'
result_name = 'circle_gtr_9_7_ronghe.csv'

is_list = []

if __name__ == '__main__':
    set_seed()

    config = CfgNode.load_cfg(open('./config/config.yaml'))
    all_dl = load_client_data(config.data_sp)

    all_client=[]

    val_error = []

    for i in range(1,14):
        client_id = i
        client_name = 'client' + str(i)
        client_cfg = config[client_name]
        client_train_dl = all_dl[i]['train']
        client_val_dl = all_dl[i]['val']
        client_test_dl = all_dl[i]['test']
        use_edge = True
        client_cfg.writer_path =client_tensorboard_path
        use_double = False
        use_res=False
        alpha = False
        focal = False

        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,use_res=False,use_teacher=False,use_gtr=True)
        all_client.append(client)
        # client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,alpha=alpha,use_focal=focal,double=True,use_teacher=False,use_dg=True,use_pgnn=True)
        # client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl,use_e=True,double=use_double,use_dg2=True)


        model_path = model_file_path + str(i) + '/best.pt'
        client.load_model_by_path(model_path)
        
        error,_ = client.val(client_val_dl)
        val_error.append(error)

        print('Client:{} Error:{}'.format(i,error))
        
        client.result(result_name, result_path)

    res = socre(base_error,val_error)
    print(res)
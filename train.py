'''
    guozhenyuan
    2022-8-5
'''

import enum
from re import I
from client import Client
from utils import load_client_data
from yacs.config import CfgNode

from server import FedBN, fedAvg

base_error = [0.263789,0.289617,0.355404,0.176471,0.396825,0.261580,0.302378,0.211538,0.059199,0.007083,0.734011,1.361326,0.004389]

def socre(base_error,val_error):
    s = 0
    for i,j in zip(base_error,val_error):
        s = (i-j)/i + s
        print("bi: {:.8f}\tval: {:.8f}\timprove_ratio: {:.8f}".format(i, j,(i-j)/i))
    return s/13


if __name__ == '__main__':

    config = CfgNode.load_cfg(open('./config.yaml'))
    all_dl = load_client_data(config.data_sp)
    all_client=[]

    for i in range(1,14):
        client_id = i
        client_name = 'client' + str(i)
        client_cfg = config[client_name]
        client_train_dl = all_dl[i]['train']
        client_val_dl = all_dl[i]['val']
        client_test_dl = all_dl[i]['test']
        client = Client(id=i,config=client_cfg,train_dl=client_train_dl,val_dl=client_val_dl,test_dl=client_test_dl)
        all_client.append(client)

    for r in range(1):
        print('round:',r)
        all_weight = []
        val_error = []
        for i,client in enumerate(all_client):
            weight = client.update(all_client)
            all_weight.append(weight)
            val_error.append(client.best_error)
        res = socre(base_error,val_error)
        print(res)
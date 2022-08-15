''' 
2022-8-6
guozhenyuan
'''

import torch

def fedAvg(clients_weight):
    tmp = {}

    # 将提取特征部分的均值保存下来
    for key,value in clients_weight[0].items():
        if 'feature' in key:

            tmp[key] = torch.mean(torch.stack([cw[key] for cw in clients_weight]),dim=0)

    new_clients_weight=[]
    for cw in clients_weight:
        tmp_cw = {}
        for key,value in cw.items():
            if key in tmp.keys():
                print('okk')
                tmp_cw[key] = tmp[key]
            else:
                tmp_cw[key] = value
        new_clients_weight.append(cw)

    return new_clients_weight                

def FedBN(cws):
    new_cws=[]

    avg_cw_no_norms = {}
    for name,value in cws[0].items():
        if 'norms' not in name and 'gnn' in name: # 当不是batchnorms并且是gnn时候进行均值聚合
            avg_cw_no_norms[name] = torch.mean(torch.stack([cw[name] for cw in cws]),dim=0)
    
    for cw in cws:
        tmp = {}
        for name, value in cw.items():
            if name in avg_cw_no_norms.keys():
                tmp[name] = avg_cw_no_norms[name]
            else:
                tmp[name] = value
        new_cws.append(tmp)
    
    return new_cws


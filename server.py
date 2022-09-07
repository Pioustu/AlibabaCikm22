''' 
2022-8-6
guozhenyuan
'''
import enum
from operator import le
from tkinter import N
import numpy as np
import math
import torch
from model import Alpha

from new_model import GIN_Net,GINE_Net

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


def FedBN_GNN(cws):  # 由于网络没有特殊BN层，所以只对卷积层进行均值即可
    new_cws=[]

    avg_cw_no_norms = {}
    for name,value in cws[0].items():
        if 'conv' in name: # 当不是batchnorms并且是gnn时候进行均值聚合
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

def FedBNGnn(gnns,lin=False):
    new_gnns = []

    # 求出不在norms层其他层中的均值保存在avg_gnns_no_norms中
    avg_gnns_no_norms = {}
    for name,value in gnns[0].items():
        if 'norms' not in name:
            avg_gnns_no_norms[name] = torch.mean(torch.stack([gnn[name] for gnn in gnns]),dim=0)
    
    # 构建norms的值
    for gnn in gnns:
        tmp = {}
        for name, value in gnn.items():
            if name in avg_gnns_no_norms.keys(): # 当在不是norms层的时候使用均值
                tmp[name] = avg_gnns_no_norms[name]
            else:  # 当不在norms层的时候使用本身的值
                tmp[name] = value
        new_gnns.append(tmp)
    
    return new_gnns

def FedBNGnn2(gnns):
    new_gnns = []

    # 求出不在norms层其他层中的均值保存在avg_gnns_no_norms中
    avg_gnns_no_norms = {}
    for name,value in gnns[0].items():
        if 'linears' in name or 'eps' in name:
            avg_gnns_no_norms[name] = torch.mean(torch.stack([gnn[name] for gnn in gnns]),dim=0)
    
    # 构建norms的值
    for gnn in gnns:
        tmp = {}
        for name, value in gnn.items():
            if '.lin.' in name: # 因为gnn_c是不带edge_attr的，所以这里可以直接跳过
                # print('kk')
                continue
            if name in avg_gnns_no_norms.keys(): # 当在不是norms层的时候使用均值
                tmp[name] = avg_gnns_no_norms[name]
            else:  # 当不在norms层的时候使用本身的值
                tmp[name] = value
        new_gnns.append(tmp)
    
    return new_gnns

def FedBn(cws,avg_list=[]):
    new_cws = []
    for i in range(len(cws)):
        tmp =  {}
        for name,value in cws[i].items():
            if name in avg_list:
                tmp[name] = torch.mean(torch.stack([cw[name] for cw in cws]),dim=0)
            else:
                tmp[name] = value
        new_cws.append(tmp)
    return new_cws

def FedBnAp(cws,avg_list=[],p=0.7,grouped=False):
    bnmlist = []
    bnvlist = []
    for cw in cws:
        tmp_bnm = []
        tmp_bnv = []
        for key,value in cw.items():
            if 'running_mean' in key:
                tmp_bnm.append(value.detach().to('cpu').numpy())
            if 'running_var' in key:
                tmp_bnv.append(value.detach().to('cpu').numpy())
        bnmlist.append(tmp_bnm)
        bnvlist.append(tmp_bnv)
    #获取权重矩阵
    M = get_weight_matrix1(bnmlist,bnvlist,model_momentum=p)
    new_cws = []
    for i in range(len(cws)):
        tmp =  {}
        for name,value in cws[i].items():
            if name in avg_list:
                tmp[name] = sum([M[i][j]*cws[j][name] for j in range(len(cws))])             
            else:
                tmp[name] = value
        new_cws.append(tmp)
    return new_cws


def FedBN(cws,lin=False): 
    new_cws=[]

    avg_cw_no_norms = {}
    for name,value in cws[0].items():
        if 'encoder' not in name or '.lin.' not in name or 'norm' not in name:# 当不是batchnorms并且是gnn时候进行均值聚合
            # print(name)
            avg_cw_no_norms[name] = torch.mean(torch.stack([cw[name] for cw in cws]),dim=0)
        # if lin == True and 'linear' in name:
        #     avg_cw_no_norms[name] = torch.mean(torch.stack([cw[name] for cw in cws]),dim=0)
    
    for cw in cws:
        tmp = {}
        for name, value in cw.items():
            if name in avg_cw_no_norms.keys():
                tmp[name] = avg_cw_no_norms[name]
            else:
                tmp[name] = value
        new_cws.append(tmp)
    
    return new_cws

def get_wasserstein(m1, v1, m2, v2, mode='nosquare'):
    w = 0
    bl = len(m1)
    for i in range(bl):
        tw = 0
        tw += (np.sum(np.square(m1[i]-m2[i])))
        tw += (np.sum(np.square(np.sqrt(v1[i]) - np.sqrt(v2[i]))))
        if mode == 'square':
            w += tw
        else:
            w += math.sqrt(tw)
    return w


def get_weight_matrix1(bnmlist, bnvlist, model_momentum=0.7):
    client_num = len(bnmlist)
    weight_m = np.zeros((client_num, client_num))
    for i in range(client_num):
        for j in range(client_num):
            if i == j:
                weight_m[i, j] = 0
            else:
                tmp = get_wasserstein(
                    bnmlist[i], bnvlist[i], bnmlist[j], bnvlist[j])
                if tmp == 0:
                    weight_m[i, j] = 100000000000000
                else:
                    weight_m[i, j] = (1/tmp)
    weight_s = np.sum(weight_m, axis=1)
    weight_s = np.repeat(weight_s, client_num).reshape(
        (client_num, client_num))
    weight_m = (weight_m/weight_s)*(1-model_momentum)
    for i in range(client_num):
        weight_m[i, i] = model_momentum
    return weight_m


def FedCos(cws,fed_layer):
    new_cws = []
    for i in range(len(cws)):
        tmp =  {}
        for name,value in cws[i].items():
            if name in fed_layer:
                all_cos = torch.stack([torch.cosine_similarity(value.view(-1), cws[j][name].view(-1), dim=0, eps=1e-08) for j in range(len(cws))],dim=0)
                mean_all_cos = torch.mean(all_cos)
                alpha = torch.where(all_cos>0,torch.ones_like(mean_all_cos),torch.zeros_like(mean_all_cos))
                tmp[name] = torch.mean(torch.stack([alpha[j]*cws[j][name] for j in range(len(cws))]),dim=0)
            else:
                tmp[name] = value
        new_cws.append(tmp)
    return new_cws

def FedAP_Plus(cws,gnn=True,linear=True,clf=True):
    # 获取所有客户机的所有bn层m和v
    bnmlist = []
    bnvlist = []
    for cw in cws:
        tmp_bnm = []
        tmp_bnv = []
        for key,value in cw.items():
            if 'running_mean' in key:
                tmp_bnm.append(value.detach().to('cpu').numpy())
            if 'running_var' in key:
                tmp_bnv.append(value.detach().to('cpu').numpy())
        bnmlist.append(tmp_bnm)
        bnvlist.append(tmp_bnv)
    #获取权重矩阵
    M = get_weight_matrix1(bnmlist,bnvlist)
    new_cws=[]
    print(M)
    for i in range(len(cws)):
        tmp = {}
        for name,value in cws[i].items():
            if 'encoder' in name or '.lin.' in name or 'norm' in name:
                tmp[name] = value
            else:
                tmp[name] = sum([M[i][j]*cws[j][name] for j in range(len(cws))])
        new_cws.append(tmp)
    
    return new_cws


def FedBN_Att(cws): 
    
    # 初始化new_cws,否则会被覆盖
    new_cws=[]
    for i in range(100):
        tmp = {}
        for key in cws[0].keys():
            tmp[key] = None
        new_cws.append(tmp)


    for name,value in cws[0].items():
        # if 'norms' not in name and ('gnn' in name or 'linear' in name): # 当不是batchnorms并且是gnn或者linea时候进行均值聚合
        if 'norms' not in name and 'gnn' in name: # 当不是batchnorms并且是gnn时候进行均值聚合
            all_value = torch.stack([cw[name] for cw in cws])
            if 'weight' in name:
                n,h,w = all_value.shape 
                scale = (h*w)** -0.5
                f_all_value = all_value.view(n,-1)
                # att = torch.mm(f_all_value,f_all_value.T)*scale
                att = torch.cosine_similarity(f_all_value.unsqueeze(1),f_all_value.unsqueeze(0),dim=2)*scale
                att = torch.softmax(att,dim=-1)
                result = torch.einsum('n n, n i j -> n i j',att,all_value)
            elif 'bias' in name:
                n,h = all_value.shape
                scale = h**-0.5
                f_all_value = all_value.view(n,-1)
                # att = torch.mm(f_all_value,f_all_value.T)
                att = torch.cosine_similarity(f_all_value.unsqueeze(1),f_all_value.unsqueeze(0),dim=2)*scale
                att = torch.softmax(att,dim=-1)
                result = torch.einsum('n n, n i -> n i',att,all_value)
            else:
                result = all_value
                print(name)
                print('eps')
        else:
            result = [cw[name] for cw in cws]
            if 'clf.weight' in name :
                print('kk',result[0].shape)
            # print([cw[name].shape for cw in cws])
        
        for i in range(len(result)):
            new_cws[i][name] = result[i] 
        
        # if 'clf.weight' in name:
        #     print('tt0',new_cws[0]['clf.weight'].shape)
        #     print('tt1',new_cws[1]['clf.weight'].shape)

    return new_cws



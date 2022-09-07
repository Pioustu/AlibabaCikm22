import enum
import numpy as np
import math
import os
import torch

from utils import set_seed

source_path = '/home/featurize/data/cikm2022/CIKM22Competition/'
target_path = '/home/featurize/cikm22/data/full/'

train_data_client_path = [source_path + str(i) + '/train.pt' for i in range(1,14)]
train_data_client_all = [torch.load(p) for p in train_data_client_path]

val_data_client_path = [source_path + str(i) + '/val.pt' for i in range(1,14)]
val_data_client_all = [torch.load(p) for p in val_data_client_path]

test_data_client_path = [source_path + str(i) + '/test.pt' for i in range(1,14)]
test_data_client_all = [torch.load(p) for p in test_data_client_path]

set_seed()
id = 0
for train,val,test in zip(train_data_client_all,val_data_client_all,test_data_client_all):
    id = id + 1
    if id <=8:# 分类
        P = [data for data in train if data.y==1]
        N = [data for data in train if data.y==0]
        num_p,num_n = len(P),len(N)
        if num_p>num_n: # 正样本大于负样本，重复采样负样本
            while True:
                N = N + N
                if len(N)>len(P):
                    break
            N = N[:len(P)]
        else: # 负样本大于正样本，重复采样
            while True:
                P = P + P
                if len(P)>len(N):
                    break
            P = P[:len(N)]
        PN = P + N
        np.random.shuffle(PN)
        for i,d in enumerate(PN):
            d.data_index = i
        new_P = [data for data in PN if data.y==1]
        new_N = [data for data in PN if data.y==0]
        # print('Clinet:{} P:{} N:{}'.format(id,len(new_P),len(new_N)))
    else:
        PN = train
    dir = target_path + '/CIKM22Competition/' + str(id) 
    if not os.path.exists(dir):
        os.makedirs(dir)
    path_train = dir + '/train.pt'
    path_val = dir + '/val.pt'
    path_test = dir + '/test.pt'
    torch.save(PN,path_val)
    torch.save(val,path_train)
    torch.save(test,path_test)

    



# id = 1
# for train,val,test in zip(train_data_client_all,val_data_client_all,test_data_client_all):
#     dataset = train+val
#     if id <=8: # 分类正负样本分别划分，之后在进行拼接保证5Fold正负样本对称
#         P = [data for data in dataset if data.y==1]
#         N = [data for data in dataset if data.y==0]
#         print(len(P),len(N))
#         P_N = len(P)
#         N_N = len(N)
#         P_idxs = list(range(P_N))
#         N_idxs = list(range(N_N))
#         P_bin = math.ceil(P_N/5)
#         N_bin = math.ceil(N_N/5)
#         Fold_P = []
#         Fold_N = []
#         for i in range(4): # 首先获取四个，最后一个用于补齐
#             tmp = []
#             P_rdm = np.random.choice(P_idxs,P_bin,replace=False)
#             N_rdm = np.random.choice(N_idxs,N_bin,replace=False)
#             P_tmp = [P[k] for k in P_rdm]
#             N_tmp = [N[k] for k in N_rdm]
#             Fold_P.append(P_tmp)
#             Fold_N.append(N_tmp)
#             P_idxs = list(set(P_idxs)-set(P_rdm))
#             N_idxs = list(set(N_idxs)-set(N_rdm))
#         P_tmp = [P[k] for k in P_idxs]
#         N_tmp = [N[k] for k in N_idxs]
#         Fold_P.append(P_tmp)
#         Fold_N.append(N_tmp)
#         Fold = [p+n for p,n in zip(Fold_P,Fold_N)]
#         print(len([i for f in Fold_P for i in f]),len([i for f in Fold_N for i in f]),len(Fold))
#     else: #回归直接均匀划分即可
#         N = len(dataset)
#         T_idxs = list(range(N))
#         T_bin = math.ceil(N/5)
#         Fold = []
#         for i in range(4):
#             tmp = []
#             rdm = np.random.choice(T_idxs,T_bin,replace=False)
#             tmp = [dataset[k] for k in rdm]
#             Fold.append(tmp)
#             T_idxs = list(set(T_idxs)-set(rdm))
#         tmp = [dataset[k] for k in T_idxs]
#         Fold.append(tmp)
#         print(N,len([i for f in Fold for i in f]))
    
#     # 打散列表
#     # np.random.shuffle

#     # 分别保存四个划分结果，并且对data_index进行重新初始化,并且保存到相应的位置
#     for k in range(len(Fold)):
#         F_val = Fold[k]
#         F_train = Fold[(k+1)%5] + Fold[(k+2)%5] + Fold[(k+3)%5] + Fold[(k+4)%5]
#         # 打乱
#         np.random.shuffle(F_train)
#         np.random.shuffle(F_val)
#         # data_index生成
#         for i,d in enumerate(F_val):
#             d.data_index = i
#         for i,d in enumerate(F_train):
#             d.data_index = i
        
#         # 保存路径
#         dir = target_path + 'Fold' + str(k+1) + '/CIKM22Competition/' + str(id) 
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#         path_train = dir + '/train.pt'
#         path_val = dir + '/val.pt'
#         path_test = dir + '/test.pt'
#         torch.save(F_val,path_val)
#         torch.save(F_train,path_train)
#         torch.save(test,path_test)

#     id = id + 1
#     # print(len(P),len(N),len(P)//5,len(N)//5)
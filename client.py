'''
    by:GuoZhenyuan
    time: 2022-8-3
'''
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from FocalLoss import WeightedFocalLoss
from model import GCN, GCN_C,GCN_R,GIN_R,GIN_C, GNN

from models.graph_level import GNN_Net_Graph
from tensorboardX import SummaryWriter


class Client(object):
    def __init__(self,id,config,train_dl,val_dl,test_dl=None):
        self.id = id
        self.round=0
        self.cfg = config

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model().to(self.device)

        self.error = 0
        self.best_error = 99999
        self.best_loss = 99999
        # 加载数据
        self.tra_dl = train_dl
        self.val_dl = val_dl
        self.tes_dl = test_dl

        self.writer = SummaryWriter('./runs_logs/'+str(self.id))
    
        # 定义损失函数
        if self.cfg.task == 'C': # 执行分类任务
            self.criterion = torch.nn.CrossEntropyLoss()
            # focal loss
            # self.criterion = WeightedFocalLoss()
            

        elif self.cfg.task == 'R':
            self.criterion = torch.nn.MSELoss()
        else:
            print('输入的任务类型有误')
        
        # 定义优化器
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=config.lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.cfg.lr,weight_decay=0.0005)

    def load_model(self):
        dim_edge = self.cfg.dim_edge if self.cfg.dim_edge > 0 else None
        
        if self.cfg.task == 'C':
            model = GNN_Net_Graph(in_channels=self.cfg.in_dim,
                out_channels=self.cfg.num_cls,
                hidden=64,
                max_depth=3,
                dropout=.3,
                gnn='gin',
                # pooling='att',
                pooling='mean',
                use_edge=False,
                edge_dim=dim_edge)
        else:
            model = GNN_Net_Graph(in_channels=self.cfg.in_dim,
                out_channels=self.cfg.num_cls,
                hidden=128,
                max_depth=3,
                dropout=.2,
                gnn='gin',
                # pooling='att',
                pooling='mean',
                use_edge=False,
                edge_dim=dim_edge)
        
        # 加载向右训练一轮后的gnn模型
        # model_weight = torch.load('./result/model_8.14_turnright/'+str(self.id)+'/best.pt')
        # tmp = {}
        # for key, value in model_weight.items():
        #     if 'gnn' in key:
        #         # print(key)
        #         tmp[key[4:]] = value
        # model.gnn.load_state_dict(tmp)
        # print("加载全局模型")

        return model

    def model_update_by_server(self,weight):
        self.model.load_state_dict(weight)

    def load_model_by_path(self,path):
        model_dict = torch.load(path, map_location="cuda:0")
        self.model.load_state_dict(model_dict)

    # def update(self,val=False,test=False):
    def update(self,all_client, val=False,test=False):
        acc = 0
        mse = 99999999999999
        loss = 9999999999999

        # 循环替换gnn,从第二个客户端开始
        # if self.id > 1:
        #     self.model.gnn = all_client[self.id - 1].model.gnn
        #     print("GNN replace OK")

    
        for e in range(1,self.cfg.ep+1):
            # print('Start Client Update')

            if self.cfg.task == "C":
                self.train_C()
                self.test_C()
                # save model during best acc
                if self.val_acc > acc:
                    acc = self.val_acc
                    self.saveModel()
                    print(acc,"save model")
                    self.best_error = self.error
                
                # save model during lowest loss
                # if self.val_loss < loss:
                #     loss = self.val_loss
                #     self.saveModel()
                #     print(loss,"save model")
                #     self.best_loss = self.val_loss

            else:
                self.train_R()
                self.test_R()
                # save model during best mse
                if self.val_mse < mse:
                    mse = self.val_mse
                    self.saveModel()
                    print(mse,"save model")
                    self.best_error = self.error


            self.printInf()
            self.writer.add_scalar('error', self.error, e)
        
        return self.model.state_dict()

    def train_C(self):
        self.model.train()
        acc = 0 # 分类时候的正确率
        los = 0

        for i, data in enumerate(self.tra_dl):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            out = self.model(data)

            # 计算损失
            label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
            # print(out.shape,label.shape)
            loss = self.criterion(out,label)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 计算正确率和损失
            pred = out.argmax(dim=1)
            acc += int((pred == data.y.squeeze(1)).sum())
            los += loss.item()
        acc = acc / len(self.tra_dl.dataset)
        los = los / len(self.tra_dl.dataset)

        self.train_acc,self.train_los = acc,los
        # print('Client:{} Train C loss: {:.5f} acc: {:.5f}'.format(self.id,los,acc))
    
    def train_R(self):
        self.model.train()
        mse = 0

        for i, data in enumerate(self.tra_dl):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            out = self.model(data)

            # 计算损失
            label = data.y
            loss = self.criterion(out,label)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 激素啊mse
            mse += loss.item()
        self.trian_mse = mse/len(self.tra_dl)
        # print('Client:{} Train R MSE:{:.5f}'.format(self.id,mse))

    def test_C(self):
        self.model.eval()
        acc = 0 # 分类时候的正确率
        los = 0

        for i, data in enumerate(self.val_dl):

            data = data.to(self.device)
            out = self.model(data)

            # 计算损失
            label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
            loss = self.criterion(out,label)

            # 计算正确率和损失
            pred = out.argmax(dim=1)
            acc += int((pred == data.y.squeeze(1)).sum())
            los += loss.item()
        acc = acc / len(self.val_dl.dataset)
        los = los / len(self.val_dl.dataset)
        self.val_acc,self.val_loss = acc,los
        self.error = 1-acc
        # print('Client:{} Val C loss: {:.5f} acc: {:.5f}'.format(self.id,los,acc))
    
    def test_R(self):
        self.model.eval()
        mse = 0
        for i, data in enumerate(self.val_dl):
            data = data.to(self.device)
            out = self.model(data)

            # 计算损失
            label = data.y
            loss = self.criterion(out,label)

            # mse
            mse += loss.item()
        
        self.val_mse=mse/len(self.val_dl)
        self.error = self.val_mse
        # print('Client:{} Val R MSE:{:.5f}'.format(self.id,mse))

    def printInf(self):
        if self.cfg.task == "C":
            print('Client: {:2d} \t Task: C \t Train ACC: {:.5f} \t Tran Loss: {:.5f} \t Val ACC: {:.5f} \t Val Loss: {:.5f}'.format(self.id,self.train_acc,self.train_los,self.val_acc,self.val_loss))
        else:
            print('Client: {:2d} \t Task: R \t Train MSE: {:.5f} \t Val MSE: {:.5f} '.format(self.id,self.trian_mse,self.val_mse))

    def result(self):
        self.model.eval()
        result = []
        data_index = []
        path = './data/cikm22/'
        data_index = [self.tes_dl.dataset[i].data_index for i in range(len(self.tes_dl.dataset))]
        print(data_index)
        for i, data in enumerate(self.tes_dl):
            data = data.to(self.device)
            # print(len(data.data_index),data.data_index)
            out = self.model(data)
            if self.cfg.task=='C':
                pred = out.argmax(dim=1)
            else:
                pred = out
            
            pred = pred.detach().cpu().numpy()
            index = data.data_index.detach().cpu().numpy()
            result = result + list(pred)
            # data_index = data_index + list(index)

        with open(os.path.join(path, 'prediction.csv'), 'a') as file:
            for y_ind, y_pred in zip(data_index, result):
                if self.cfg.task == 'C':
                    line = [self.id, y_ind] + [y_pred]
                else:
                    line = [self.id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')
    
    def saveModel(self):
        path = './result/model/'+ str(self.id) 
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/best.pt'
        torch.save(self.model.state_dict(), path)


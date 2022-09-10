'''
    by:GuoZhenyuan
    time: 2022-8-3
'''

import copy
import os
import torch
from torch._C import device
import torch.nn.functional as F
from new_model import GTR

from tensorboardX import SummaryWriter
from utils import FocalLoss

share_layers = [
    # 'gnn.convs.0.eps',
    'gnn.convs.0.nn.linears.0.weight',
    'gnn.convs.0.nn.linears.0.bias',
    'gnn.convs.0.nn.linears.1.weight',
    'gnn.convs.0.nn.linears.1.bias',
    # 'gnn.convs.1.eps',
    'gnn.convs.1.nn.linears.0.weight',
    'gnn.convs.1.nn.linears.0.bias',
    'gnn.convs.1.nn.linears.1.weight',
    'gnn.convs.1.nn.linears.1.bias',
    'gnn.convs.2.nn.linears.0.weight',
    'gnn.convs.2.nn.linears.0.bias',
    'gnn.convs.2.nn.linears.1.weight',
    'gnn.convs.2.nn.linears.1.bias'
]


def softmax(p_list):
    kk = sum([torch.exp(i) for i in p_list])
    tmp = [torch.exp(i)/kk for i in p_list]
    return tmp

class Client(object):
    def __init__(self,id,config,train_dl,val_dl,test_dl=None,use_e=False,alpha=None,use_focal=False,\
        use_teacher=True,double=False,use_res=False,use_dg=False,use_pgnn=False,use_gtr=False):

        self.id = id
        self.round=0
        self.update_num = 0
        self.cfg = config

        self.use_dg = use_dg

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 判断是否使用边
        if use_e==False or self.cfg.edge_dim==0:
            use_edge = False
            print('Not Use Edge')
        else:
            use_edge = True
            print('Use Edge')
        
        self.double = double
        # 加载自己模型和teacher模型
        self.model = self.load_model(use_edge,double,use_res,use_dg=use_dg,use_pgnn=use_pgnn,use_gtr=use_gtr).to(self.device)
        if use_teacher == True:
            self.t_model = self.load_model(use_edge,double,use_res).to(self.device)

        # 加载数据
        self.tra_dl = train_dl
        self.val_dl = val_dl
        self.tes_dl = test_dl

        # 是否使用focal函数
        self.focal = use_focal

        # 定义损失函数
        if self.cfg.task == 'C': # 执行分类任务
            if self.focal == True:
                self.criterion = FocalLoss(num_class=2,gamma=3,alpha=alpha)
            else:
                self.criterion = torch.nn.CrossEntropyLoss()
        elif self.cfg.task == 'R':
            self.criterion = torch.nn.MSELoss()
            # 为了数据标准化，因此我们计算一下均值方差让标签的分布都变成N（mean,var），可以同理转换为minMSE（label - mean）/ var - output 所以，我们只需要变换一下minMSE(label - output*var+mean)
            # 因此只用对输出进行反向标准化即可
            self.var = torch.var(torch.stack([d.y for d in train_dl.dataset ]))
            self.mean = torch.mean(torch.stack([d.y for d in train_dl.dataset ]))
        else:
            print('输入的任务类型有误')

        # 保存初始化时候的error和best_error
        self.error = 0
        self.best_error,_ = self.val(val_dl)
        # self.best_error = 9999999
        print('Client:{} Start Best Error:{}'.format(self.id,self.best_error))

        if use_dg == False:
            self.best_gnn_weight = copy.deepcopy(self.model.gnn.state_dict())
        
        # 初始化prox模型参数值
        self.prox_weight = None
        
        # 输出tensorboard保存路径
        self.writer = SummaryWriter(self.cfg.writer_path + 'client' + str(self.id))

    def load_model(self,use_e=False,double=False,use_res=False,use_dg=False,use_pgnn=False,use_gtr=False): # 用于加载模型，use_e代表是否使用边
        '''
            用于初始化模型
            输入：
                use_e: 使用边
                double: 使用双gnn网络
                use_res: gnn网络中添加残差连接
                use_dg: 使用doubel gnn网络
                use_pgnn: 使用个性化gnn网络
                use_gtr: 使用构建的GTR模型 注：最终使用的是这个模型
            输出：
                model: 经过选择的模型

        '''
        if use_gtr == True:
            model = GTR(
                in_channels=self.cfg.in_dim,
                out_channels=self.cfg.num_cls,
                hidden=128,
                max_depth=3,
                dropout=self.cfg.dropout,
                use_edge=use_e,
                edge_dim=self.cfg.edge_dim
            )
        return model

    def model_update_by_server(self,weight): # 通过weight加载模型
        self.model.load_state_dict(weight)
    
    def get_model_weight(self,weight):
        tmp = {}
        # 只要不带norms的层
        for key, value in self.model.state_dict().items(): 
            if 'gnn' in key:
                if 'norms' in key or '.lin.' in key: # 使用lin能够达到49？？？
                    tmp[key] = value
                else:
                    tmp[key] = weight[key]
            else:
                tmp[key] = value
        return tmp


    def load_fed(self,weights,lr,epoch):

        opt = torch.optim.Adamax(self.alpha.parameters(),lr=lr)
        
        # 获取13个模型的输出值
        outputs=[]
        for weight in weights:
            self.model.load_state_dict(self.get_model_weight(weight))
            out = self.val(dl=self.tra_dl,op=True)
            outputs.append(self.val(dl=self.tra_dl,op=True))
        self.alpha.train()
        for e in range(epoch):
            losses = 0
            for i, data in enumerate(self.tra_dl):
                data = data.to(self.device)
                out = self.alpha([output[i] for output in outputs])
                # print(out.shape)
                # print(data.y.shape)
                if self.cfg.task=='C': # 判断分类回归，focalloss来选择标签值
                    label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
                    if self.focal:
                        label = data.y
                else:
                    label = data.y
                loss = self.criterion(out,label)
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses = losses + loss.item()
            print('Epoch:{} Alpha:{} Loss:{}'.format(e,torch.softmax(self.alpha.alpha,dim=0),losses))
        print('Client:{} Alpha:{}'.format(self.id,torch.softmax(self.alpha.alpha,dim=0)))    
        tmp = {}
        for name,value in self.model.state_dict().items():
            alpha = torch.softmax(self.alpha.alpha,dim=0)
            if name in share_layers:
                tmp[name] =  sum([alpha[i]*weight[name] for i, weight in enumerate(weights)])
                # print(tmp[name])
            else:
                tmp[name] = value
        self.model.load_state_dict(tmp)
                
    def load_model_by_path(self,path): # 通过保存的路径加载模型
        model_dict = torch.load(path, map_location="cuda:0")
        # tmp = {}
        # for key,value in self.model.state_dict().items():
        #     if key in model_dict.keys():
        #         tmp[key] = model_dict[key]
        #         # print(key,tmp[key].shape,value.shape)
        #     else:
        #         tmp[key] = value
        self.model.load_state_dict(model_dict,strict=True)

    def load_self_best_gnn(self):
        self.model.gnn.load_state_dict(self.best_gnn_weight)


    def load_mate(self,p_weight,keep_pgnn):
        tmp={}
        for name, value in self.model.state_dict().items():
            if name in p_weight.keys() and 'norm' not in name:
                if keep_pgnn==True and 'pgnn' in name:
                    # print(name)
                    tmp[name] = value
                elif p_weight[name].shape == value.shape:
                    tmp[name] = p_weight[name]
                else:
                    tmp[name] = value
            else:
                tmp[name] = value
        
        self.model.load_state_dict(tmp)

    def load_p_weight(self,p_weight,layer_list):
        '''
            加载前一个客户端的模型参数，并且将共享层参数替换到当前客户端模型中
            输入：
                p_weight: 表示前一个模型的所有参数值，字典格式使用model.state_dict()提取
                layer_list: 表示需要共享的层名称
            输出：
                无

        '''
        tmp =  {}
        for name,value in self.model.state_dict().items():
            if name in layer_list:
                tmp[name] = p_weight[name]
            else:
                tmp[name] = value
        self.model.load_state_dict(tmp)

    def load_gnn(self,p_weight): # 只加载gnn层，使用fedBN方式加载
        tmp = {}
        # 只要不带norms的层
        for key, value in self.model.state_dict().items(): 
            if 'gnn' in key:
                if 'norms' in key or '.lin.' in key: # 使用lin能够达到49？？？
                    tmp[key] = value
                else:
                    tmp[key] = p_weight[key]
            else:
                tmp[key] = value
        
        self.model.load_state_dict(tmp,strict=False) # 只把不带norms的层进行替换

    # def load_gnnc_only(self,gnn_c_weight):
    #     tmp = {}
    #     for key,value in self.model.gnn_c.state_dict().items():
    #         if 'norms'

    def load_linear(self,p_weight):
        tmp = {}
        # 只要不带norms的层
        for key, value in self.model.state_dict().items(): 
            if 'linear' in key:
                tmp[key] = p_weight[key]
            else:
                tmp[key] = value
        
        self.model.load_state_dict(tmp,strict=False) # 只把不带norms的层进行替换
    
    def update(self,all_layer=False,encoder=True,gnn=True,gtr=True,pgnn=False,ggnn=False,ggnn_bn=False,linear=True,plinear=False,glinear=False,clf=True,\
        stage=None,kd=False,prox=False,alpha=False):

        '''
            用于更新client的模型参数, 将client中模型参数根据数据进行更新
            输入：
                all_layer: True时候表示更新所有参数
                encoder: True时候表示更新模型的encoder模块
                gnn: True时候表示更新模型的gnn模块
                gtr: True时候表示更新模型的gtr模块
                pgnn: True时候表示更新模型的pgnn模块
                gnn_bn: True时候表示更新gnn的bn层模块
                linear: True时候表示更新模型的linear模块
                plinear: True时候表示更新plinear模块
                glinear: True时候表示更新glinear模块
                clf: True时候表示更新clf模块
                stage: 不同的阶段，用于标识不同的阶段，控制不同的学习率和轮数
                kd: 是否使用知识蒸馏方法
                alpha: 表示是否使用focalloss中的alpha参数
            输出：
                无
        '''

        if prox == True:
            self.model_weight = copy.deepcopy(self.model.state_dict())

        if all_layer == False:

            # 判断训练那一层，将其添加到update_layer中
            update_layer = []
            if encoder == True:
                update_layer.append({'params':self.model.encoder.parameters()})
            if gnn == True:
                update_layer.append({'params':self.model.gnn.parameters()})
            if pgnn == True:
                update_layer.append({'params':self.model.pgnn.parameters()})
            if ggnn == True:
                update_layer.append({'params':self.model.ggnn.parameters()})
            if ggnn_bn == True:
                for name,param in self.model.ggnn.named_parameters():
                    if 'norm' in name:
                        param.requires_grad = True
                        update_layer.append({'params':param})
            if linear == True:
                update_layer.append({'params':self.model.linear.parameters()})
            if plinear == True:
                update_layer.append({'params':self.model.plinear.parameters()})
            if glinear == True:
                update_layer.append({'params':self.model.glinear.parameters()})
            if alpha == True:
                update_layer.append({'params':self.alpha})
            if clf == True:
                update_layer.append({'params':self.model.clf.parameters()})
            if gtr == True:
                update_layer.append({'params':self.model.gtr.parameters()})
        

        if stage== 'is':
            # print('is')
            lr = self.cfg.is_lr
            epoch = self.cfg.is_ep
            if self.cfg.task == 'C':
                is_no_update_stop = 300
            else:
                is_no_update_stop = 100
        elif stage == 'ci':
            # print('ci')
            lr = self.cfg.ci_lr
            epoch = self.cfg.ci_ep
        elif stage == 'ft':
            # print('ft')
            lr = self.cfg.ft_lr
            epoch = self.cfg.ft_ep
            ft_no_update_stop = 100
        elif stage == 'double':
            lr = self.cfg.lr
            epoch = self.cfg.ep
        elif stage == 'norm':
            lr = self.cfg.lr
            epoch = self.cfg.ep
        else:
            print('输入正确stage')

        if all_layer == True:
            opt = torch.optim.Adamax(self.model.parameters(),lr=lr)
        else:
            opt = torch.optim.Adamax(update_layer,lr=lr)

        for e in range(epoch):
            if stage == 'ft':
                if ft_no_update_stop == 0:
                    print('FT No Imporve SO Stop')
                    break
                ft_no_update_stop -= 1
            if stage == 'is':
                if is_no_update_stop<=0:
                    print('IS No Imporve SO Stop')
                    break
                is_no_update_stop = is_no_update_stop - 1
            train_error,train_loss = self.train(opt,kd,prox) # 开始训练
            val_error,val_loss = self.val(self.val_dl) # 验证当前模型
            
            # 输出相关信息
            print('Client:{:2d} Round:{} Epoch:{} Train Error:{:.5f} Val Error:{:.5f} Train Loss:{:.5f} Val Loss:{:.5f}'.format(self.id,self.round,e,train_error,val_error,train_loss,val_loss))
            
            
            # 保存error到client中
            self.error = val_error
            if self.best_error > val_error: # 出现比当前最优模型更好的模型时候执行的操作
                self.best_error = val_error
                # self.best_gnn_weight = copy.deepcopy(self.model.gnn.state_dict())
                print('Client:{:2d} Round:{} Epoch:{} Save Model Best Error:{}'.format(self.id,self.round,e,self.best_error))
                self.saveModel() # 保存最优模型

                if stage == 'ft': # 更新后计数重启
                    ft_no_update_stop = 100
                if stage == 'is': # 更新后计数重启
                    if self.cfg.task == 'C':
                        is_no_update_stop = 300
                    else:
                        is_no_update_stop = 100
                
                # 更新最好模型参数
                self.best_weight = copy.deepcopy(self.model.state_dict)
            
            # 更新self.update_num,绘图
            self.writer.add_scalar('Train Error',train_error, self.update_num)
            self.writer.add_scalar('Val Error',val_error,self.update_num)
            self.update_num = self.update_num + 1
            
        self.round = self.round + 1

    def train(self,opt,kd=False,prox=False): # 正常的训练
        '''
            用于模型的训练
            输入：
                opt: 一个优化器，里面包含了模型需要更新的模块
                kd: 是否使用知识蒸馏
                prox: 是否使用prox正则约束
            输出：
                train_err: 训练集的误差
                train_los: 训练集的损失
        '''
        self.model.train()

        train_err = 0
        train_acc = 0
        train_los = 0
        # outputs = []

        if kd==True: # 知识蒸馏
            teacher_error,_ = self.val(self.val_dl,teacher=True)
            now_error,_ = self.val(self.val_dl)

            if teacher_error<=now_error: # 动态蒸馏参数调整
                lam = 2 * 10**(min(1,(now_error-teacher_error)*5)-1)
            else:
                lam = 0

        for i, data in enumerate(self.tra_dl):
            opt.zero_grad()
            data = data.to(self.device)
            out = self.model(data)
            
            if self.cfg.task=='C': # 判断分类回归，focalloss来选择标签值
                label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
                if self.focal:
                    label = data.y
            else:
                label = data.y
            loss = self.criterion(out,label)
            
            if kd == True: # 如果进行知识蒸馏时候loss添加l2loss
                t_out = self.t_model(data)
                t_loss = 0.5* torch.norm(out-t_out,p=2)
                loss = (1-lam)*loss + lam*t_loss
            if prox == True: # prox正则约束
                for key,value in self.model.named_parameters():
                    l2_loss = torch.pow(torch.norm(value - self.prox_weight[key]), 2)
                    loss = loss + (0.002 / 2.) * l2_loss
                    # print(l2_loss.item())
            loss.backward()
            opt.step()
            
            if self.cfg.task=='C':
                pred = out.argmax(dim=1)
                train_acc += int((pred == data.y.squeeze(1)).sum())
            train_los += loss.item()
        
        if self.cfg.task == 'C':
            train_acc = train_acc / len(self.tra_dl.dataset)
            train_los = train_los / len(self.tra_dl)
            train_err = 1 - train_acc
        else:
            train_los = train_los / len(self.tra_dl)
            train_err = train_los
        
        # outputs = torch.stack(outputs)
        
        return train_err, train_los


    def val(self,dl,op=False,teacher=False): # 正常的验证，返回error
        '''
            验证函数
            输入：
                dl: 需要验证的数据集
                op: 是否输出模型的输出
                teacher: 是否对teacher模型进行验证
            输出：
                op == True: 输出模型的output
                op != True: 输出模型的误差和验证数据的损失
        '''
        model = self.model.eval()

        acc = 0
        los = 0
        outputs = []
        # 如果是teacher模型就用teacher进行验证
        if teacher == True:
            model = self.t_model.eval()

        for i, data in enumerate(dl):

            data = data.to(self.device)
            # print(len(data.data_index),data.data_index)
            out = model(data)
            # print(out.shape)
            outputs.append(out.detach().to(self.device))
            if self.cfg.task=='C':
                label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
                if self.focal:
                    label = data.y
            else:
                label = data.y    

            # 计算正确率和损失
            loss = self.criterion(out,label)
            if self.cfg.task=='C':
                pred = out.argmax(dim=1)
                acc += int((pred == data.y.squeeze(1)).sum())
            los += loss.item()

        if self.cfg.task == 'C':
            acc = acc / len(dl.dataset)
            los = los / len(dl)
            err = 1 - acc
        else:
            los = los / len(dl)
            err = los
        if op == True:
            return outputs
        return err,los

    def per_val(self,dl,teacher=False):
        # 模型验证
        acc = 0
        los = 0
        model = self.model.eval()
        if teacher == True:
            model = self.t_model.eval()

        for i, data in enumerate(dl):

            data = data.to(self.device)
            # print(len(data.data_index),data.data_index)
            out = model(data)
            if self.cfg.task=='C':
                label = F.one_hot(data.y,num_classes=2).squeeze(1).float()
                if self.focal:
                    label = data.y
            else:
                label = data.y    

            # 计算正确率和损失
            loss = self.criterion(out,label)
            if self.cfg.task=='C':
                pred = out.argmax(dim=1)
                acc += int((pred == data.y.squeeze(1)).sum())
            los += loss.item()

        if self.cfg.task == 'C':
            acc = acc / len(dl.dataset)
            err = 1 - acc
        else:
            err = los / len(dl)
        
        return err

    def result(self,name,path):
        '''
            保存结果到path/name.csv下面
            输入：
                name: 结果的名字
                path: 结果的路径
            输出：
                无
        '''

        self.model.eval()
        result = []
        data_index = [self.tes_dl.dataset[i].data_index for i in range(len(self.tes_dl.dataset))]
        # print(data_index[:5])
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
        #     data_index1 = data_index + list(index)
        # print(data_index1[:5])

        with open(os.path.join(path, name), 'a') as file:
            for y_ind, y_pred in zip(data_index, result):
                if self.cfg.task == 'C':
                    line = [self.id, y_ind] + [y_pred]
                else:
                    line = [self.id, y_ind] + list(y_pred)
                file.write(','.join([str(_) for _ in line]) + '\n')
    
    def saveModel(self):
        '''
            保存模型
        '''
        self.cfg.save_path
        path = self.cfg.save_path + str(self.id) 
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + '/best.pt'
        torch.save(self.model.state_dict(), path)


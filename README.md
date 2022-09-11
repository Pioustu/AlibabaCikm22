# CIKM 2022 AnalytiCup Competition: 联邦异质任务学习

# Team: 我又来垫底了

## Code Structure
~~~
AlibabaCikm22:
    ├── cikm2022
    │   └── CIKM22Competition    # dataset
    ├── circle_gtr.py
    ├── client.py
    ├── config
    │   └── config.yaml
    ├── ft.py
    ├── new_model.py
    ├── README.md
    ├── result
    ├── server.py
    ├── test.py
    └── utils.py
~~~

## 1、快速开始
我们为官方组织者提供了复现结果的步骤。

### Step 1. 安装
我们根据说明安装并配置了运行环境: [Step-by-step Guidance for CIKM 2022 AnalytiCup Competition](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.0.0.47943ab4wXx3Ts&postId=402279)

### Step 2. 准备数据集
在运行`AlibabaCikm22`之前，应该先下载官方提供的[数据集](https://tianchi.aliyun.com/competition/entrance/532008/information)。之后将解压的文件(CIKM22Competition)放在`AlibabaCikm22/cikm22/`目录下。

### Step 3. Make some modifications
为了提高性能，我们只修改了很少的代码。你可以找到次文件: `anaconda3/envs/fs/lib/python3.9/site-packages/torch_geometric/nn/conv/gin_conv.py`,并做如下修改。
~~~python
# change line 145
# before change
if edge_dim is not None:
    if hasattr(self.nn[0],'in_features'):
        in_channels = self.nn[0].in_features
    else:
        in_channels = self.nn[0].in_channels
    self.lin = Linear(edge_dim, in_channels)
# after change
if edge_dim is not None:
    if hasattr(self.nn.linears[0], 'in_features'):
        in_channels = self.nn.linears[0].in_features
    elif hasattr(self.nn[0],'in_features'):
        in_channels = self.nn[0].in_features
    else:
        in_channels = self.nn[0].in_channels
    self.lin = Linear(edge_dim, in_channels)
~~~

### Step 4. 运行
~~~ python
cd AlibabaCikm22

python circle_gtr.py
~~~

### Step 5. 测试
~~~ python
python test.py 
~~~

## 2、算法概览
我们的算法包含了两个部分：模型结构和训练策略
1. 模型结构：我们使用`pyG`作为我们的模型搭建库进行搭建，使用`GINConv,GINEConv`作为我们`GNN`的基础节点和边的特征提取模块，使用`GraphMultisetTransformer`作为我们图的表示模块生成图特征，之后使用线性层进行分类和回归
2. 训练策略: 我们使用环状的方式进行训练，第`i-1`客户端的模型参数会传递给第`i`个客户端，第`i`个客户端使用第`i-1`个客户端的参数进行学习，依次类推形成了一个环状的结构进行训练

## 3、联邦学习细节
我们的联邦学习策略主要借鉴了`FedPer,FedBN,FedProx,MetaFed`四个个主要的算法

模型结构：`Encoder`（将节点特征映射到`GNN`输入特征），`GNN`（图特征提取，输出图的特征表示），`Decoder`（线性映射，将图的特征表示映射到不同输出纬度），其中`Encoder`和`Decoder`作为私有层不进行参数共享，`GNN`层作为共享层用于联邦的训练

`GNN`共享层细节：借鉴`FedBN`中的策略，我们`GNN`中的共享时候不会对`BN`层和`LN`层进行共享，只共享`Linear`层。由于部分数据存在边，这导致`GNN`中会多出来一个边的映射层，这一层也不进行参数共享，从而我们的GNN层只共享大家都有的层。

训练策略：我们借鉴`MateFed`中的第一阶段中`circle`的训练策略，使用环形的训练方式，即第`i`个客户端首先获取第`i-1`个客户端的模型，在第`i-1`个模型上使用第`i`个客户端的数据进行训练，之后将训练好的模型传递给第`i+1`个客户端，形成了一个环状结构。

`prox`策略：为了模型更新的方向偏差太大，我们借鉴了`FedProx`的正则约束方式，将第`j`个客户端的模型作为正则约束项，在使用第`i-1`个客户端的模型进行训练时候进行约束。

## 4、数据传输细节
我们使用了自己构建联邦学习的方式来完成这次比赛，其中数据的读取、结果的生成和部分模型的搭建使用了`FS`中的部分代码。

数据传递我们使用`torch.model.state_dict()`来获取模型的参数，将参数字典作为我们的信息传递给下一个客户端，通过`torch.model.load_state_dict()`来对参数进行加载。

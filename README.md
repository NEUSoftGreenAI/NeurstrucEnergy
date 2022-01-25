# NeurstrucEnergy

|-- NeurstrucEnergy   
　　|-- .DS_Store   
　　|-- .gitignore   
　　|-- LICENSE   
　　|-- README.md   
　　|-- utlis.py   
　　|-- CNNGeneration   
　　|　　|-- CustomCNNGeneration.py   
　　|　　|-- RandomCNNGeneration.py   
　　|-- data   
　　|　　|-- .DS_Store   
　　|　　|-- conv2d_energy_profiling.py   
　　|　　|-- energy_profiling.py   
　　|　　|-- Monitor.py   
　　|　　|-- dataset   
　　|　　　　|-- .DS_Store   
　　|　　　　|-- conv2d.txt   
　　|　　　　|-- conv2d_PreVIousNet.txt   
　　|　　　　|-- custom_testset_1.txt   
　　|　　　　|-- custom_testset_2.txt   
　　|　　　　|-- custom_testset_3.txt   
　　|　　　　|-- NNtest.py   
　　|　　　　|-- random_testset_1.txt   
　　|　　　　|-- random_testset_2.txt   
　　|　　　　|-- random_testset_3.txt   
　　|-- model   
　　|　　|-- .DS_Store   
　　|　　|-- BiGNN.py   
　　|　　|-- train.py   
　　|　　|-- Transformer.py   
　　|　　|-- PerLayerRegression   
　　|　　　　|-- NeuralPower_PolyModel.py   
　　|　　　　|-- PreVIous_LinearModel.py   
　　|-- TrainedModel   
　　　　|-- .DS_Store   
　　　　|-- CustomPredict.py   
　　　　|-- RandomPredict.py   
　　　　|-- CustomModel   
　　　　|　　|-- .DS_Store   
　　　　|　　|-- model_592   
　　　　|-- RandomModel   
　　　　　　 |-- model_432   


### File Description 文件说明


File | Description
---|---
utlis.py | Some functions utilized by other codes. （共用函数）
CustomCNNGeneration.py | Customized generation algorithm （订制生成算法）
RandomCNNGeneration.py | Random generation algorithm（随机生成算法）
conv2d\_energy\_profiling.py | Obtain inference energy consumption of conv2d layer. （得到conv2d层的推理能耗）
energy\_profiling.py | Obtain inference energy consumption of networks. （得到神经网络的推理能耗）
Monitor.py | Connect with the power meter. （与功率计通信）
conv2d.txt | 1000 convolution layers from the randomly generated dataset. （从随机生成数据集中抽取的1000个卷积层）
conv2d_PreVIousNet.txt | Conv2d layers dataset utilized by PreVIous. （PreVIous的Conv2d层数据集） 
custom\_testset\_*.txt | The 5 test nets for the experiment on custom dataset. （定制数据集实验中的5个测试网络）
random\_testset\_*.txt | The 17 test nets for the experiment on random dataset. （随机数据集实验中的17个测试网络）
BiGNN.py | An implemention of bi-directional GNN. （双向GNN实现）
train.py | Train our bi-directional GNN model. （训练模型）
Transformer.py | An implemention of Transformer. （Trainsformer实现）
NeuralPower_PolyModel.py | The experiment on polynomial regression model for layer regression. （多项式模型预测层能耗实验）
PreVIous_LinearModel.py | The experiment on linear regression model for layer regression. （线性回归模型预测层能耗实验）
CustomPredict.py | The experiment on custom dataset. （定制数据集实验）
RandomPredict.py | The experiment on random dataset. （随机数据集实验）
model_592 | The parameters of model trained on custom dataset. （定制数据集模型参数）
model_432 | The parameters of model trained on random dataset. （随机数据集模型参数）


### Model Training 训练模型


##### Trained on randomly generated dataset


```
python train.py
```


##### Trained on custom dataset


```
python train.py --hidden 80 --reverse-hidden 16 --lr 0.00045 --dataset custom
```

### Reproduce the experimental results 复现实验结果

We only provide the parameters of bi-directional model for one epoch in this depository, and the download addresses for all 50 consecutive epochs and the complete dataset are as follows:

在本仓库中我们只提供训练过程中1轮的模型参数，全部连续50轮的模型参数和完整数据集的下载地址如下：

> https://drive.google.com/drive/folders/1THfcWXFy_EFYByZPDoRLth7sMspQaLEM?usp=sharing

> 链接: https://pan.baidu.com/s/1FTFTu9ocspmFkU6_1QabtA 提取码: jfli

##### Trained on randomly generated dataset


```
python RandomPredict.py
```


##### Trained on custom dataset


```
python CustomPredict.py
```


import time
import torch
import torch.nn as nn

class NN_chain(nn.Module):
    def __init__(self):
        super(NN_chain,self).__init__()
        self.conv_list = [nn.Conv2d(64,64,1,1,0) for i in range(51)]
        self.x_list = [0 for i in range(51)]
        
    def forward(self,x):
        self.x_list[0] = self.conv_list[0](x)
        for i in range(1,51):
            self.x_list[i] = self.conv_list[i](self.x_list[i-1])
        return x

class NN_DAG(nn.Module):
    def __init__(self):
        super(NN_DAG,self).__init__()
        self.conv_list = [nn.Conv2d(64,64,1,1,0) for i in range(51)]
        self.x_list = [0 for i in range(51)]
    def forward(self,x):
        self.x_list[0] = self.conv_list[0](x)
        for i in range(1,51):
            self.x_list[i] = self.conv_list[i](self.x_list[0])
        return x

inp = torch.rand(1,64,28,28)
NNchain = NN_chain()
NNDAG = NN_DAG()
mean_time = 0
for i in range(1000):
    start = time.time()
    otp = NNchain(inp)
    mean_time += time.time()-start
print('NNchain',mean_time/1000)
mean_time = 0
for i in range(1000):
    start = time.time()
    otp = NNDAG(inp)
    mean_time += time.time()-start
print('NNDAG',mean_time/1000)
mean_time = 0
for i in range(1000):
    start = time.time()
    otp = NNchain(inp)
    mean_time += time.time()-start
print('NNchain',mean_time/1000)
mean_time = 0
for i in range(1000):
    start = time.time()
    otp = NNDAG(inp)
    mean_time += time.time()-start
print('NNDAG',mean_time/1000)

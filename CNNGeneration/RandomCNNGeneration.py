import os
from threading import Thread
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from multiprocessing import Process
from numpy import *
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager
from queue import Queue
import collections
import random
import requests
import csv
import string
import pandas as pd
import math
import traceback
# from monitor import Monitor
# from stableMonitor import stableMonitor
np.set_printoptions(threshold=500)
       
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7,10) #10分类的问题
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.out(x)
        return x

def achieve_stable_energy(return_dict):
    '''
    由于要尽量避免GPU的一些例如温度、静默功率、峰值功率等对后续的计算造成的影响，先运行一段时间等到功耗稳定后再进行后续生成
    '''
    monitor = stableMonitor(0.01)
    EPOCH = 1000
    BATCH_SIZE = 1000
    LR = 0.001

    x = torch.rand(BATCH_SIZE,1,28,28)
    b_x = Variable(x).cuda()
    cnn = CNN()
    cnn.cuda()
    optimizer = optim.Adam(cnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    '''
    训练COUNT_TIME轮后，每前向传播一次计算一次方差，连续EARLY_STOP_TIME次方差降低，则继续，否则结束
    '''
    EARLY_STOP_TIME = 2
    COUNT_TIME = 10
    energy_cost_std_list = np.zeros(EARLY_STOP_TIME + 1)
    count = 0
    while(True):
      torch.cuda.synchronize()
      monitor.begin()
      for j in range(0,30):
        print(j)
        output = cnn(b_x)
      count += 1
      torch.cuda.synchronize()
      monitor.stop()
      time.sleep(2)

      if(count >= COUNT_TIME):
        data = monitor.get_stable_energy_list()
        for i in range(0,EARLY_STOP_TIME):
          energy_cost_std_list[i] = energy_cost_std_list[i+1]
        energy_cost_std_list[EARLY_STOP_TIME] = std(data)
        print("方差列表：",energy_cost_std_list)
        for i in range(0,EARLY_STOP_TIME):
          if(energy_cost_std_list[i]!=0 and energy_cost_std_list[i]<energy_cost_std_list[i+1]):
            return_dict['stable_silence_value'] = monitor.get_silence_energy()
            print("能耗校正完成！")
            monitor.exit()
            return
          else:
            break

class NNgenerator(nn.Module):
    def __init__(self,layer_parameters,layer_link,layer_id):
        super(NNgenerator,self).__init__()
        self.layer_parameters = layer_parameters
        self.layer_link = layer_link
        self.layer_id = layer_id
        self.layer_list = []
        self.parameters_flag = 0
        self.link_flag = 0
        # print(len(self.layer_id))
        self.link_graph = self.link_vector_to_graph(self.layer_link,len(self.layer_id))
        # print(self.link_graph)
        in_degree_list = self.get_in_degree()
        out_degree_list = self.get_out_degree()
        # print(in_degree_list)
        # print(out_degree_list)
        for i in range(0,len(self.layer_id)):
          params_length = self.get_params_length(self.layer_id[i])
          link_length = self.get_link_length(i)
          self.layer_list.append(self.make_layer(self.layer_parameters[self.parameters_flag:self.parameters_flag+params_length], self.layer_link[self.link_flag:self.link_flag+link_length], self.layer_id[i]))
          self.parameters_flag += params_length
          self.link_flag += link_length
        self.layer_list = nn.ModuleList(self.layer_list)
        # print(self.layer_list)
        
    def make_layer(self, parameters, link, id):
        '''
        生成一个层
        '''
        # print(parameters,id,len(parameters))
        if(id == 0):
          in_channels,out_channels,kernel_size,stride,padding,dilation,groups = parameters[-7:]
        
          return nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups)

        elif(id == 1):
          # print(parameters)
          in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups = parameters[-11:]
          
          return nn.Conv2d(in_channels,out_channels,(kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
                  padding=(padding_height, padding_width),dilation=(dilation_height, dilation_width),groups=groups)
          
        elif(id == 2):
          in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_width,dilation_height,groups = parameters[-15:]
          
          return nn.Conv3d(in_channels,out_channels,(kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),\
                  padding=(padding_depth, padding_height, padding_width),dilation=(dilation_depth, dilation_height, dilation_width),groups=groups)
          
        elif(id == 3):
          in_channels,out_channels,kernel_size,stride,padding,output_padding,dilation,groups = parameters[-8:]

          return nn.ConvTranspose1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding,dilation=dilation,groups=groups)
          
        elif(id == 4):
          in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,output_padding_height,output_padding_width,dilation,groups = parameters[-12:]
          # print(in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,output_padding_height,output_padding_width,dilation,groups)
          return nn.ConvTranspose2d(in_channels,out_channels,(kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
                  padding=(padding_height, padding_width),output_padding=(output_padding_height,output_padding_width),dilation=dilation,groups=groups)
          
        elif(id == 5):
          in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,\
          padding_height,padding_width,output_padding_depth,output_padding_height,output_padding_width,dilation,groups = parameters[-16:]
        
          return nn.ConvTranspose3d(in_channels,out_channels,(kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),\
                  padding=(padding_depth, padding_height, padding_width),output_padding=(output_padding_depth, output_padding_height, output_padding_width),dilation=dilation,groups=groups)
        
        elif(id == 6):
          #如果是max pooling则需要返回indices
          kernel_size,stride,padding,dilation,pool_type = parameters[-5:]
          if(pool_type == 0):
            #为max pooling
            return nn.MaxPool1d(kernel_size = kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)
          else:
            #为avg pooling
            #不支持dilation,在生成时要默认成1
            return nn.AvgPool1d(kernel_size = kernel_size, stride=stride, padding=padding)
                   
        elif(id == 7):
          #如果是max pooling则需要返回indices
          kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,pool_type = parameters[-9:]
          if(pool_type == 0):
            #为max pooling
            return nn.MaxPool2d(kernel_size = (kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
                   padding=(padding_height, padding_width),dilation=(dilation_height, dilation_width), return_indices=True)
          else:
            #为avg pooling
            #不支持dilation,在生成时要默认成1
            return nn.AvgPool2d(kernel_size = (kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
                   padding=(padding_height, padding_width))
            
        elif(id == 8):
          #如果是max pooling则需要返回indices
          kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_height,dilation_width,pool_type = parameters[-13:]
          if(pool_type == 0):
            #为max pooling
            return nn.MaxPool3d(kernel_size = (kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),\
                   padding=(padding_depth, padding_height, padding_width),dilation=(dilation_depth, dilation_height, dilation_width), return_indices=True)
          else:
            #为avg pooling
            #不支持dilation,在生成时要默认成1
            return nn.AvgPool3d(kernel_size = (kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),\
                   padding=(padding_depth, padding_height, padding_width))
            
        elif(id == 9):
          kernel_size,stride,padding = parameters[-3:]

          return nn.MaxUnpool1d(kernel_size = kernel_size, stride=stride, padding=padding)

        elif(id == 10):
          kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width = parameters[-6:]

          return nn.MaxUnpool2d(kernel_size = (kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),padding=(padding_height, padding_width))

        elif(id == 11):
          kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width = parameters[-9:]

          return nn.MaxUnpool3d(kernel_size = (kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),padding=(padding_depth, padding_height, padding_width))

        elif(id == 12):
          output_size_L,pool_type = parameters[-2:]
          if(pool_type == 0):
            #为max pooling return_indices
            return nn.AdaptiveMaxPool1d(output_size_L)
          else:
            #为avg pooling
            return nn.AdaptiveAvgPool1d(output_size_L)

        elif(id == 13):
          output_size_H,output_size_W,pool_type = parameters[-3:]
          if(pool_type == 0):
            #为max pooling return_indices
            return nn.AdaptiveMaxPool2d((output_size_H,output_size_W))
          else:
            #为avg pooling
            return nn.AdaptiveAvgPool2d((output_size_H,output_size_W))

        elif(id == 14):
          output_size_D,output_size_H,output_size_W,pool_type = parameters[-4:]
          if(pool_type == 0):
            #为max pooling return_indices
            return nn.AdaptiveMaxPool3d((output_size_D,output_size_H,output_size_W))
          else:
            #为avg pooling
            return nn.AdaptiveAvgPool3d((output_size_D,output_size_H,output_size_W))
        
        elif(id == 15):
          num_features = parameters[-1:][0]
          return nn.BatchNorm1d(num_features)

        elif(id == 16):
          num_features = parameters[-1:][0]
          return nn.BatchNorm2d(num_features)

        elif(id == 17):
          num_features = parameters[-1:][0]
          return nn.BatchNorm3d(num_features)
        
        elif(id == 18):
          probability = parameters[-1:][0]
          return nn.Dropout(p=probability)

        elif(id == 19):
          probability = parameters[-1:][0]
          return nn.Dropout(p=probability)
          
        elif(id == 20):
          probability = parameters[-1:][0]
          return nn.Dropout(p=probability)
          
        elif(id == 21):
          input_length,output_length = parameters[1],parameters[3]
          return nn.Linear(input_length,output_length)
          
        elif(id == 22):
          sigmoid,tanh,ReLU,leaky_ReLU = parameters[-4:]
          if(sigmoid == 1):
            return nn.Sigmoid()
          elif(tanh == 1):
            return nn.Tanh()
          elif(ReLU == 1):
            return nn.ReLU()
          else:
            return nn.LeakyReLU()
        
        #由于add和concat不是实际的神经网络层，随便返回一个神经网络层
        elif(id == 23):
          return nn.ReLU()

        elif(id == 24):
          return nn.ReLU()

        elif(id == 25):
          probability = parameters[-1:][0]
          return nn.Dropout2d(p=probability)

        elif(id == 26):
          probability = parameters[-1:][0]
          return nn.Dropout3d(p=probability)

    def forward(self,x):
        '''
        出度为0，输出return
        入度为0，表示接收初始输入
        入度为1，正常节点，第一个位置表示接收初始输入
        入度>1，concat或add节点

        实现方法：
        ① 找到入度为0元素为初始输入
        ② 广度优先遍历邻接矩阵,将计算结果保存在列表comp_context中
        ③ 返回出度为0的结果
        '''
        layer_length = len(self.layer_id)
        queue = Queue(layer_length)
        comp_context = [0 for index in range(layer_length)]
        unpool_indices = [0 for index in range(layer_length)]
        in_degree_list = self.get_in_degree()
        out_degree_list = self.get_out_degree()
        BFS_flag = np.zeros(layer_length,dtype = int)
        #广度遍历
        for i in range(0,layer_length):
          if(in_degree_list[i] == 0):
            queue.put(i)
            BFS_flag[i] = 1
        while not queue.empty():
          layer_index = queue.get()
          # print(layer_index)
          if(in_degree_list[layer_index] == 0):
            '''
            该节点入度为0时，只需要接受初始输入
            '''
            # print('self.layer_list[layer_index]',self.layer_id[layer_index])
            # print('first',self.layer_list[layer_index],layer_index)
            comp_context[layer_index] = self.layer_list[layer_index](x)
            # print('first',comp_context[layer_index].shape)
            # print(comp_context)
            children_indices = self.get_children_indices(layer_index)#找到子节点位置
            for i in range(0,len(children_indices)):
              if(BFS_flag[children_indices[i]] == 0):
                queue.put(children_indices[i])
                BFS_flag[children_indices[i]] = 1            
          elif(in_degree_list[layer_index] > 0):
            '''
            该节点入度大于0时，找到所有父节点。
            入度为1直接接受父节点输入，入度大于1为concat层和add层
            '''
            parent_indices = self.get_parent_indices(layer_index)#找到父节点位置

            '''
            如果父节点没有执行，即comp_context中没有计算结果，则将其重新放入队列尾。
            由于广度优先遍历，入度为1时不存在这个问题，入度为2时可能存在
            '''
            if(len(parent_indices) == 1):
              #全连接层需要加入一个展平操作
              # print(comp_context[parent_indices[0]].size())
              # # print(self.layer_list[layer_index])
              # print("父节点",parent_indices[0])
              # print("当前节点",layer_index)
              if(self.layer_id[layer_index] == 21):
                dimension = len(comp_context[parent_indices[0]].size())
                #维度是2，之前有全连接层，已经展平了
                if(dimension == 2):
                  # print('self.layerid',self.layer_id[layer_index])
                  comp_context[layer_index] = self.layer_list[layer_index](comp_context[parent_indices[0]])
                else:
                  # print(comp_context[parent_indices[0]].shape)
                  comp_context[layer_index] = comp_context[parent_indices[0]].view(comp_context[parent_indices[0]].size(0),-1)
                  # print(comp_context[layer_index].shape)
                  comp_context[layer_index] = self.layer_list[layer_index](comp_context[layer_index])
              else:
                if(self.layer_id[layer_index] == 6 or self.layer_id[layer_index] == 7 or self.layer_id[layer_index] == 8):
                  #池化层，考虑是否是MaxPool，如果是，加入indices
                  if(hasattr(self.layer_list[layer_index], 'return_indices')):
                    comp_context[layer_index],unpool_indices[layer_index] = self.layer_list[layer_index](comp_context[parent_indices[0]])
                    # print('保存',layer_index,'unpool_indices')
                  else:
                    # print('self.layerid',self.layer_id[layer_index])
                    comp_context[layer_index] = self.layer_list[layer_index](comp_context[parent_indices[0]])
                elif(self.layer_id[layer_index] == 9 or self.layer_id[layer_index] == 10 or self.layer_id[layer_index] == 11):
                  #反池化层，使用indices
                  # print(unpool_indices[parent_indices[0]])
                  comp_context[layer_index] = self.layer_list[layer_index](comp_context[parent_indices[0]],unpool_indices[parent_indices[0]])
                else:
                  # print('self.layerid',self.layer_id[layer_index],'父节点',parent_indices,'父节点类型',self.layer_id[parent_indices[0]])
                  comp_context[layer_index] = self.layer_list[layer_index](comp_context[parent_indices[0]])
              # print('child',comp_context[layer_index])
              children_indices = self.get_children_indices(layer_index)#找到子节点位置
              for i in range(0,len(children_indices)):
                if(BFS_flag[children_indices[i]] == 0):
                  queue.put(children_indices[i])
                  BFS_flag[children_indices[i]] = 1  

            elif(len(parent_indices) > 1):
              #检查是否存在comp_context没有计算结果的节点，如果都计算过，则执行else后的语句
              for i in range(0,len(parent_indices)):
                if(type(comp_context[parent_indices[i]]) != torch.Tensor):
                  queue.put(layer_index)
                  break
              else:
                # print(self.layer_list[layer_index])
                # print("父节点",parent_indices)
                # for i in range(0,len(parent_indices)):
                  # print(comp_context[parent_indices[i]].size())
                # print("当前节点",layer_index)
                #把多个parent输出连接成元组
                converge_tuple = ()
                for i in range(0,len(parent_indices)):
                  converge_tuple += (comp_context[parent_indices[i]],)
                if(self.layer_id[layer_index] == 23):
                  #concat层
                  comp_context[layer_index] = torch.cat(converge_tuple, 1)#channel拼接
                elif(self.layer_id[layer_index] == 24):
                  #add层
                  comp_context[layer_index] = converge_tuple[0]
                  for i in range(1,len(parent_indices)):
                    # print(comp_context[layer_index].size(),converge_tuple[i].size())
                    comp_context[layer_index] = torch.add(comp_context[layer_index],converge_tuple[i])
          
                #找到该节点的子节点加入队列
                children_indices = self.get_children_indices(layer_index)#找到子节点位置
                for i in range(0,len(children_indices)):
                  if(BFS_flag[children_indices[i]] == 0):
                    queue.put(children_indices[i])
                    BFS_flag[children_indices[i]] = 1
        
        #找到入度为0的节点return
        return_list = []
        for i in range(0,layer_length):
          if(out_degree_list[i] == 0):
            return_list.append(comp_context[i])
        
        return return_list
    
    def get_link_length(self,pos):
      '''
      获取当前id的连接向量长度
      '''
      return pos+1

    def get_in_degree(self):
      '''
      根据邻接矩阵，获取节点入度列表
      '''
      in_degree_list = []
      for i in range(0,len(self.layer_id)):
        node_row = list(self.link_graph[i])
        node_row.pop(i)
        in_degree_list.append(np.array(node_row).sum())
      return in_degree_list
    
    def get_out_degree(self):
      '''
      根据邻接矩阵，获取节点出度列表
      '''
      out_degree_list = []
      for i in range(0,len(self.layer_id)):
        #除去对角元
        node_column = list(self.link_graph[:,i])
        node_column.pop(i)
        out_degree_list.append(np.array(node_column).sum())
      return out_degree_list

    def get_parent_indices(self,index):
      '''
      找到第index个节点的依赖输出节点
      '''
      node_row = list(self.link_graph[index])
      parent_list = []
      for i in range(0,len(node_row)):
        if(node_row[i] == 1 and i != index):
          parent_list.append(i)
      return parent_list

    def get_children_indices(self,index):
      '''
      找到第index个节点的子节点（即接收其输入的节点）
      '''
      node_column = list(self.link_graph[:,index])
      children_list = []
      for i in range(0,len(node_column)):
        if(node_column[i] == 1 and i != index):
          children_list.append(i)
      return children_list

    def link_vector_to_graph(self,link_list,length):
      '''
      将连接向量转化成邻接矩阵，对角线元素表示是否接收初始输入
      '''
      graph = np.zeros([length,length],dtype = float)
      flag = 0
      if len(link_list) != length * length:
        for i in range(0,length):
          for j in range(0,i+1):
            graph[i,j] = link_list[flag]
            flag += 1
      else:
        for i in range(0,length):
          for j in range(0,length):
            graph[i,j] = link_list[flag]
            flag += 1
        
      return graph

    def get_params_length(self,layer_id):
      '''
      获取不同层参数向量长度
      '''
      get_params_length_dic = {
        0:13,
        1:19,
        2:25,
        3:14,
        4:20,
        5:26,
        6:11,
        7:17,
        8:23,
        9:9,
        10:14,
        11:19,
        12:7,
        13:9,
        14:11,
        15:4,
        16:5,
        17:6,
        18:4,
        19:5,
        20:6,
        21:4,
        22:6,
        23:3,
        24:3,
        25:5,
        26:6,
      }
      return get_params_length_dic[layer_id]
        
def get_one_energy():
  monitor = Monitor(0.001,2)
  layer_parameters = [1000,1,28,28,1000,16,28,28,1,16,5,5,1,1,2,2,1,1,1, 1000,16*28*28,0,0,1,0 ,1000,16,28,28,1000,16,14,14,2,2,2,2,0,0,1,1,0 , 1000,16,14,14,1000,32,14,14,16,32,5,5,1,1,2,2,1,1,1, 1000,32*14*14,0,0,1,0 ,1000,32,14,14,1000,32,7,7,2,2,2,2,0,0,1,1,0 ,1000,32*7*7,1000,10  ]
  layer_link = [1, 1,0, 0,1,0, 0,0,1,0, 0,0,0,1,0, 0,0,0,0,1,0, 0,0,0,0,0,1,0] # 最后1个表示接收原始输入
  layer_id = [1,22,7,1,22,7,21]
  NN = NNgenerator(layer_parameters,layer_link,layer_id)
  # print(NN)
  NN.cuda()
  x = torch.rand(1000,1,28,28)
  b_x = Variable(x).cuda()
  output = NN(b_x)
  # print(x)
  torch.cuda.synchronize()
  monitor.begin()
  for j in range(0,1000):
    output = NN(b_x)
      
  torch.cuda.synchronize()
  monitor.stop()
  time.sleep(2)

def validate_NN(vg,dim):
  # torch.cuda.empty_cache()
  NN = NNgenerator(vg.layer_parameters,vg.layer_link,vg.layer_id)
  # print(NN)
  # NN.cuda()
  if dim == 1:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2])
  elif dim == 2:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2],vg.net_input[3])
  else:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2],vg.net_input[3],vg.net_input[4])

  b_x = Variable(x)
  output = NN(b_x)
  total_params = sum(p.numel() for p in NN.parameters())
  # for p in NN.parameters():
  #     print(p.numel())
  print(f'{total_params:,} total parameters.')
  str1 = ''
  str1 += ",".join('%s' %i for i in vg.layer_parameters) + " "
  str1 += ",".join('%s' %i for i in vg.layer_link) + " "
  str1 += ",".join('%s' %i for i in vg.layer_id) + " "
  str1 += str(total_params) + " "
  str1 += str(vg.dimension) + " "
  str1 += str(vg.block_num) + " "
  str1 += str(vg.stream_num) + " "
  # str1 += cpu_name + " "
  # str1 += cpu_MHz + " "
  # str1 += cache_size + " "
  # str1 += str(processor_num) + " "
  # str1 += gpu_name + " "
  # str1 += "0" + " "
  # str1 += "0" + " "
  # str1 += "0" + " "
  # str1 += "0" + " "
  with open("test.txt","a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
    file.write(str1 + "\n")
    file.close()
  return True

def validate_NN_NVG(params,link,id,dim,input_shape):
  # torch.cuda.empty_cache()
  NN = NNgenerator(params,link,id)
  # NN.cuda()
  if dim == 1:
    x = torch.rand(input_shape[0],input_shape[1],input_shape[2])
  elif dim == 2:
    x = torch.rand(input_shape[0],input_shape[1],input_shape[2],input_shape[3])
  else:
    x = torch.rand(input_shape[0],input_shape[1],input_shape[2],input_shape[3],input_shape[4])

  b_x = Variable(x)
  # print(b_x.shape)
  output = NN(b_x)
  total_params = sum(p.numel() for p in NN.parameters())
  
class VectorGenerator():
  
    def __init__(self,dimension,block_num,stream_num,batchNorm_prob=0.5,dropout_prob=0.2,more_fc_prob=0.15,max_fc_num=2,delete_fc_prob=0.1,no_dropout = 0.5,large=1):
      '''
      dimension:数据维度
      block_num:CNN块的数量
      stream_num:几路神经网络，只能是1或2
      batchNorm_prob:卷积层后加入BatchNorm概率
      dropout_prob:卷积层、FC层后加入dropout概率
      more_fc_prob:多个FC层的概率
      max_fc_num:最大全连接层数量
      delete_fc_prob:最后不接FC的概率

      流程：
      对于每个神经网络流
      ① 进行block_num次循环生成block_num个神经网络块
      ② 对于每个神经网络块，有唯一的输入和输出，中间可以有分支
      ③ 如果是2路神经网络，在最后加一个add和concat操作
      ④ 如果不接FC层，最后一个块或者2路神经网络合并后的结果作为输出
      '''

      super(VectorGenerator, self).__init__()
      self.dimension = dimension # 1表示1d,2表示2d,3表示3d
      self.block_num = block_num
      self.stream_num = stream_num
      self.batchNorm_prob = batchNorm_prob
      self.dropout_prob = dropout_prob
      self.more_fc_prob = more_fc_prob
      self.max_fc_num = max_fc_num
      self.delete_fc_prob = delete_fc_prob
      self.no_dropout = no_dropout
      self.large = large
      if random.randint(1,100) <= no_dropout * 100:
        self.no_dropout = True
      else:
        self.no_dropout = False
      
      self.layer_num = 0 #记录当前已生成的节点数量
      self.layer_parameters = []
      self.layer_link = []
      self.layer_id = []
      self.net_input = self.get_net_input_size()
      # print('self.net_input',self.net_input)
      # self.make_net()
    
    def make_net(self):
      
      if(self.dimension == 1):
        #生成CNN网络
        stream_output_size = []
        stream_output_index = []
        for net_stream in range(0,self.stream_num):
        #生成多流神经网络的一条
          last_block_input_size = self.net_input
          last_block_index = -1
          for block in range(0,self.block_num):
            #print("netstream",net_stream," 第",block,"个block")
            #一条中的多个block
            input_batch_size,input_channels,input_length = last_block_input_size
            out_channels = 1
            if(input_channels <= 3):
              out_channels = random.randint(16,32)
            else:
              out_channels = random.randint(int(input_channels*1.8),input_channels*2)
            out_shape = self.prob_random([int(input_length*0.5),input_length,int(input_length/3)],[0.8,0.2,0.1])
            # print(input_height,out_shape)
            output_size = [input_batch_size,out_channels,out_shape]
            #block = 0 时，接收初始输入，为last_block_index = -1
            last_block_index,channels = self.make_block(last_block_input_size,output_size,last_block_index,4)
            output_size[1] = channels
            last_block_input_size = output_size
            if(block == self.block_num - 1):
              #记录每个流最后一个块的输出大小
              stream_output_size.append(output_size)
              stream_output_index.append(self.layer_num - 1)
        # print("生成流结束")
        fc_length = 0
        input_fc_size = last_block_input_size
        if(self.stream_num > 1):
          #把不同流的结果连接起来
          if(stream_output_size[0] != stream_output_size[1]):
            if(stream_output_size[0][2] > stream_output_size[1][2]):# 第一个更大一些

              #在第一个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_length = stream_output_size[0]
              output_size = stream_output_size[1]
              last_block_index = self.make_block(stream_output_size[0],stream_output_size[1],stream_output_index[0],4)
              stream_output_index[0] = (self.layer_num-1)
              stream_output_size[0] = stream_output_size[1]
              last_block_input_size = output_size
            
            else:#第二个更大一些

              #在第二个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_length = last_block_input_size
              output_size = stream_output_size[0]
              last_block_index = self.make_block(last_block_input_size,output_size,last_block_index,4)
              stream_output_index[1] = (self.layer_num-1)
              stream_output_size[1] = stream_output_size[0]
              last_block_input_size = output_size

          if(random.randint(1,100) < 30):
            #Add
            # print("流 添加Add")
            params = self.make_layer(24,stream_output_size[0],add_num=len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2]
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
          else:
            #concat
            #print("流 添加concat")
            params = self.make_layer(23,stream_output_size[0],out_channels=stream_output_size[0][1] * len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2] * len(stream_output_index)
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
            input_fc_size[1] *= len(stream_output_index)
          
          
        
        #生成全连接层
        if(random.randint(1,100) < self.delete_fc_prob * 100):
          #不接全连接层
          return 
        
        #计算前面所有层的参数数量
        before_fc_params_num = 0
        index = 0
        for i in range(len(self.layer_id)):
          length = self.get_params_length(self.layer_id[i])
          before_fc_params_num += self.get_params_num(self.layer_id[i],self.layer_parameters[index:index+length])
          index += length
        
        #获取全连接层的个数
        fc_num = 1
        linaer_layer_output_index_list = []
        while random.randint(1,100) <= 20:
          fc_num += 1
          if(fc_num == self.max_fc_num):
            break
        
        fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)
        while (fc_params_num_ratio>0.9 or fc_params_num_ratio<0.6):
          fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)

        fc_params_num = int(before_fc_params_num/(1-fc_params_num_ratio)) - before_fc_params_num
        
        #生成全连接层，全连接层参数个数占比在80%左右
        now_fc_num = 0
        input_length = input_fc_size[1]*input_fc_size[2]

        if fc_num == 1:
          batch_size = input_fc_size[0]
          output_length = int(fc_params_num / input_length)
          if random.randint(1,100) <= 50 and output_length > 1000:
            output_length = 1000
          # print(fc_params_num,input_length)
          assert output_length > 0, "找不到合适的全连接层"
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
        else:
          output_length_fc1 = 0
          output_length_fc2 = 0
          range_list = random.sample(range(10,1015),1000)
          for i in range(1000):
            output_length_fc2 = range_list[i]
            output_length_fc1 = int(fc_params_num / (input_length + output_length_fc2))
            if(input_length > output_length_fc1 and output_length_fc1 > output_length_fc2):
              break
            assert i < 999, "找不到合适的全连接层"
          batch_size = input_fc_size[0]
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length_fc1])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          params = self.make_layer(21,[batch_size,output_length_fc1],[batch_size,output_length_fc2])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1

      elif(self.dimension == 2):
        #生成CNN网络
        stay_channel_prob = 0.5
        stream_output_size = []
        stream_output_index = []
        for net_stream in range(0,self.stream_num):
        #生成多流神经网络的一条
          last_block_input_size = self.net_input
          last_block_index = -1
          for block in range(0,self.block_num):
            #print("netstream",net_stream," 第",block,"个block")
            #一条中的多个block
            input_batch_size,input_channels,input_height,input_width = last_block_input_size
            out_channels = 1
            out_shape = 0
              # 生成2 3 4整数倍的out channels
            if block < 5:
              out_shape = self.prob_random([int(input_height*0.5),int(input_height/3)],[0.9,0.1])
              if(input_channels <= 3):
                out_channels = random.randint(16,32)
              else:
                out_channels = random.randint(int(input_channels*2),input_channels*4)
                if(random.randint(1,100) < 30):
                  out_channels = self.prob_random([input_channels,input_channels*2,input_channels*3,input_channels*4,input_channels*6],[0.2,0.4,0.15,0.1,0.05])
                if out_channels > 1000:
                  out_channels = self.prob_random([int(input_channels/4),int(input_channels/2),out_channels],[0.3,0.5,0.2])
                #使最终out_channels期望稳定
                  if(random.randint(1,100) < 20):
                    out_channels = input_channels
              if out_shape < 7:
                out_shape = input_height
                    
            else:
              out_shape = self.prob_random([int(input_height*0.5),input_height],[0.1,0.9])
              
              if(input_channels <= 300):
                out_channels = self.prob_random([input_channels,input_channels*2,input_channels*3],[0.3,0.5,0.2])
              else:
                out_channels = self.prob_random([int(input_channels/2),input_channels,input_channels*2],[0.4,0.4,0.2])
              if out_channels > 1000:
                out_channels = self.prob_random([int(input_channels/4),int(input_channels/2),out_channels],[0.3,0.5,0.2])
              #使最终out_channels期望稳定
                if(random.randint(1,100) < 20):
                  out_channels = input_channels
              if out_shape < 7:
                out_shape = input_height
            
            # print(input_height,out_shape)
            output_size = [input_batch_size,out_channels,out_shape,out_shape]
            # print('output_size',output_size)
            #block = 0 时，接收初始输入，为last_block_index = -1
            last_block_index,channels = self.make_block(last_block_input_size,output_size,last_block_index,4)
            output_size[1] = channels
            last_block_input_size = output_size
            if(block == self.block_num - 1):
              #记录每个流最后一个块的输出大小
              stream_output_size.append(output_size)
              stream_output_index.append(self.layer_num - 1)
        # print("生成流结束")
        fc_length = 0
        input_fc_size = last_block_input_size
        if(self.stream_num > 1):
          #把不同流的结果连接起来
          if(stream_output_size[0] != stream_output_size[1]):
            if(stream_output_size[0][2] > stream_output_size[1][2]):# 第一个更大一些

              #在第一个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_height,input_width = stream_output_size[0]
              output_size = stream_output_size[1]
              last_block_index = self.make_block(stream_output_size[0],stream_output_size[1],stream_output_index[0],4)
              stream_output_index[0] = (self.layer_num-1)
              stream_output_size[0] = stream_output_size[1]
              last_block_input_size = output_size
            
            else:#第二个更大一些

              #在第二个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_height,input_width = last_block_input_size
              output_size = stream_output_size[0]
              last_block_index = self.make_block(last_block_input_size,output_size,last_block_index,4)
              stream_output_index[1] = (self.layer_num-1)
              stream_output_size[1] = stream_output_size[0]
              last_block_input_size = output_size

          if(random.randint(1,100) < 30):
            #Add
            # print("流 添加Add")
            params = self.make_layer(24,stream_output_size[0],add_num=len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2] * stream_output_size[0][3]
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
          else:
            #concat
            #print("流 添加concat")
            params = self.make_layer(23,stream_output_size[0],out_channels=stream_output_size[0][1] * len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2] * stream_output_size[0][3] * len(stream_output_index)
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
            input_fc_size[1] *= len(stream_output_index)
          
          
        
        #生成全连接层
        if(random.randint(1,100) < self.delete_fc_prob * 100):
          #不接全连接层
          return 
        
        input_length = input_fc_size[1]*input_fc_size[2]*input_fc_size[3]
        
        if input_length > 10000:
          #加一个adaptiveavgpool层
          out_shape = random.randint(1,3)
          final_output_size = [input_fc_size[0],input_fc_size[1],out_shape,out_shape]
          input_ada_size = input_fc_size
          input_fc_size = final_output_size
          # print('adaptive',final_output_size)
          params = self.make_layer(13,input_ada_size,final_output_size)
          output_size = params[4:8]
          last_layer_input_size = output_size
          input_fc_size = output_size
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [13] # 加入层id列表中
          self.layer_num += 1
        
        #计算前面所有层的参数数量
        before_fc_params_num = 0
        index = 0
        for i in range(len(self.layer_id)):
          length = self.get_params_length(self.layer_id[i])
          before_fc_params_num += self.get_params_num(self.layer_id[i],self.layer_parameters[index:index+length])
          index += length
        
        #获取全连接层的个数
        fc_num = 1
        linaer_layer_output_index_list = []
        while random.randint(1,100) <= 10:
          fc_num += 1
          if(fc_num == self.max_fc_num):
            break
        
        # fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)
        # while (fc_params_num_ratio>0.9 or fc_params_num_ratio<0.6):
        #   fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)

        # fc_params_num = int(before_fc_params_num/(1-fc_params_num_ratio)) - before_fc_params_num
        
        
        
        # #生成全连接层，全连接层参数个数占比在80%左右
        # now_fc_num = 0
        input_length = input_fc_size[1]*input_fc_size[2]*input_fc_size[3]
        # print('input_fc_size',input_fc_size)
        if fc_num == 1:
          batch_size = input_fc_size[0]
          
          output_length = random.randint(50,2000)
          if(random.randint(1,100) < 10):
            output_length = 1000
          # print(fc_params_num,input_length)
          # assert output_length > 0, "找不到合适的全连接层"
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
        else:
          output_length_fc1 = 0
          output_length_fc2 = 0
          range_list = random.sample(range(10,1015),1000)
          for i in range(1000):
            output_length_fc2 = random.randint(50,2000)
            output_length_fc1 = random.randint(500,2000)
          batch_size = input_fc_size[0]
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length_fc1])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          params = self.make_layer(21,[batch_size,output_length_fc1],[batch_size,output_length_fc2])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1

        # for i in range(0,fc_num):
        #   # print("全连接层的上一层",input_fc_size)
        #   batch_size = input_fc_size[0]
        #   if(i == fc_num - 1):
        #     output_length = random.randint(10,1000)
        #   params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length])
        #   self.layer_parameters += params #加入参数向量中
        #   link_list=[self.layer_num-1]
        #   self.layer_link += self.get_link_vector(link_list,self.layer_num)
        #   self.layer_id += [21] # 加入层id列表中
        #   self.layer_num += 1
        #   input_length = output_length
        #   output_length = int(output_length / random.randint(10,50))

      else:
        #生成CNN网络 3维
        stream_output_size = []
        stream_output_index = []
        for net_stream in range(0,self.stream_num):
        #生成多流神经网络的一条
          last_block_input_size = self.net_input
          last_block_index = -1
          for block in range(0,self.block_num):
            #print("netstream",net_stream," 第",block,"个block")
            #一条中的多个block
            input_batch_size,input_channels,input_depth,input_height,input_width = last_block_input_size
            out_channels = 1
            if(input_channels <= 3):
              out_channels = random.randint(32,64)
            else:
              out_channels = random.randint(int(input_channels*2),input_channels*4)
            output_depth = self.prob_random([int(input_depth*0.5),input_depth],[0.3,0.7])
            out_shape = self.prob_random([int(input_height*0.5),input_height,int(input_height/3)],[0.8,0.2,0.1])
            # print(input_height,out_shape)
            output_size = [input_batch_size,out_channels,output_depth,out_shape,out_shape]
            #block = 0 时，接收初始输入，为last_block_index = -1
            last_block_index,channels = self.make_block(last_block_input_size,output_size,last_block_index,4)
            output_size[1] = channels
            last_block_input_size = output_size
            if(block == self.block_num - 1):
              #记录每个流最后一个块的输出大小
              stream_output_size.append(output_size)
              stream_output_index.append(self.layer_num - 1)
          # print('stream_output_size',stream_output_size)
        # print("生成流结束")
        fc_length = 0
        input_fc_size = last_block_input_size
        if(self.stream_num > 1):
          #把不同流的结果连接起来
          if(stream_output_size[0] != stream_output_size[1]):
            if(stream_output_size[0][2] > stream_output_size[1][2]):# 第一个更大一些

              #在第一个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_depth,input_height,input_width = stream_output_size[0]
              output_size = stream_output_size[1]
              last_block_index = self.make_block(stream_output_size[0],stream_output_size[1],stream_output_index[0],4)
              stream_output_index[0] = (self.layer_num-1)
              stream_output_size[0] = stream_output_size[1]
              last_block_input_size = output_size
            
            else:#第二个更大一些

              #在第二个流上再加一个块，以匹配尺寸大小
              input_batch_size,input_channels,input_depth,input_height,input_width = last_block_input_size
              output_size = stream_output_size[0]
              last_block_index = self.make_block(last_block_input_size,output_size,last_block_index,4)
              stream_output_index[1] = (self.layer_num-1)
              stream_output_size[1] = stream_output_size[0]
              last_block_input_size = output_size

          # print('stream_output_size',stream_output_size)
          if(random.randint(1,100) < 30):
            #Add
            # print("流 添加Add")
            params = self.make_layer(24,stream_output_size[0],add_num=len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2] * stream_output_size[0][3] * stream_output_size[0][4]
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
          else:
            #concat
            #print("流 添加concat")
            params = self.make_layer(23,stream_output_size[0],out_channels=stream_output_size[0][1] * len(stream_output_index))
            fc_length = stream_output_size[0][1] * stream_output_size[0][2] * stream_output_size[0][3] * stream_output_size[0][4] * len(stream_output_index)
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(stream_output_index,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            input_fc_size = stream_output_size[0]
            input_fc_size[1] *= len(stream_output_index)
          
          
        
        #生成全连接层
        if(random.randint(1,100) < self.delete_fc_prob * 100):
          #不接全连接层
          return 
        
        #计算前面所有层的参数数量
        before_fc_params_num = 0
        index = 0
        for i in range(len(self.layer_id)):
          length = self.get_params_length(self.layer_id[i])
          before_fc_params_num += self.get_params_num(self.layer_id[i],self.layer_parameters[index:index+length])
          index += length
        
        #获取全连接层的个数
        fc_num = 1
        linaer_layer_output_index_list = []
        while random.randint(1,100) <= 20:
          fc_num += 1
          if(fc_num == self.max_fc_num):
            break
        
        fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)
        while (fc_params_num_ratio>0.9 or fc_params_num_ratio<0.6):
          fc_params_num_ratio = np.random.normal(loc=0.8,scale=0.05,size=1)

        fc_params_num = int(before_fc_params_num/(1-fc_params_num_ratio)) - before_fc_params_num
        
        #生成全连接层，全连接层参数个数占比在80%左右
        now_fc_num = 0
        input_length = input_fc_size[1]*input_fc_size[2]*input_fc_size[3]*input_fc_size[4]

        if fc_num == 1:
          batch_size = input_fc_size[0]
          output_length = int(fc_params_num / input_length)
          # print('fc_params_num,input_length,before_fc_params_num',fc_params_num,input_length,before_fc_params_num)
          assert output_length > 0, "找不到合适的全连接层"
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
        else:
          output_length_fc1 = 0
          output_length_fc2 = 0
          range_list = random.sample(range(10,1015),1000)
          for i in range(1000):
            output_length_fc2 = range_list[i]
            output_length_fc1 = int(fc_params_num / (input_length + output_length_fc2))
            if(input_length > output_length_fc1 and output_length_fc1 > output_length_fc2):
              break
            assert i < 999, "找不到合适的全连接层"
          batch_size = input_fc_size[0]
          params = self.make_layer(21,[batch_size,input_length],[batch_size,output_length_fc1])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          params = self.make_layer(21,[batch_size,output_length_fc1],[batch_size,output_length_fc2])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1

    def make_block(self,input_size,output_size,last_block_index,max_branch_layer,branch_prob = 0.1):
      '''
      input_size接收上一层输入的尺寸
      output_size输出尺寸
      max_branch_layer每一个分支最多的层数
      生成一个神经网络块，该块中的层只与块中的其他层有联系，不与其他块有联系，块中不包含FC层。
      规则:
      ①卷积层、批标准化、激活层、池化层一般遵循如下顺序：
      nn.Conv2d(1, 6, 3, padding=1),
      nn.BatchNorm2d(6),
      nn.ReLU(True),
      nn.MaxPool2d(2, 2)
      但是前后可以接多个Conv，Pool
      ②如果是concat，channel可以不一样。如果是add，所有shape都要一样。
      
      branch_prob表示生成一个分支的概率为branch_prob，生成两个为branch_prob*branch_prob，以此类推
      '''
      # print("生成block...",input_size,output_size)
      channels = output_size[1]#用来记录合并后的channels
      branch_num = 1
      linaer_layer_output_index_list = []
      while random.randint(1,100) <= branch_prob*100:
        branch_num += 1
      
      if(self.dimension == 1):
        
        for branch_index in range(0,branch_num):
          linaer_layer_output_index = self.make_linear_layers_1d(last_block_index,input_size,output_size)
          linaer_layer_output_index_list.append(linaer_layer_output_index)
        
        if(branch_num > 1):
          #连接起来
          if(random.randint(1,100) < 30):
            #Add
            #print("添加Add")
            params = self.make_layer(24,output_size,add_num=len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            
          else:
            #concat
            # print("添加concat")
            params = self.make_layer(23,output_size,out_channels=output_size[1] * len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            channels = channels * len(linaer_layer_output_index_list)
      
      elif(self.dimension == 2):
        
        if branch_num == 1 and input_size[1] > 256 and random.randint(1,100) <= 50:
          linaer_layer_output_index = self.make_Bottleneck_layers_2d(last_block_index,input_size,output_size)
          linaer_layer_output_index_list.append(linaer_layer_output_index)
          
        for branch_index in range(0,branch_num):
          linaer_layer_output_index = self.make_linear_layers_2d(last_block_index,input_size,output_size)
          linaer_layer_output_index_list.append(linaer_layer_output_index)
        
        if(branch_num > 1):
          #连接起来
          if(random.randint(1,100) < 30):
            #Add
            #print("添加Add")
            params = self.make_layer(24,output_size,add_num=len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            
          else:
            #concat
            # print("添加concat")
            params = self.make_layer(23,output_size,out_channels=output_size[1] * len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            channels = channels * len(linaer_layer_output_index_list)

      else:
        for branch_index in range(0,branch_num):
          linaer_layer_output_index = self.make_linear_layers_3d(last_block_index,input_size,output_size)
          linaer_layer_output_index_list.append(linaer_layer_output_index)
        
        if(branch_num > 1):
          #连接起来
          if(random.randint(1,100) < 30):
            #Add
            #print("添加Add")
            params = self.make_layer(24,output_size,add_num=len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [24] # 加入层id列表中
            self.layer_num += 1
            
          else:
            #concat
            # print("添加concat")
            params = self.make_layer(23,output_size,out_channels=output_size[1] * len(linaer_layer_output_index_list))
            self.layer_parameters += params #加入参数向量中
            self.layer_link += self.get_link_vector(linaer_layer_output_index_list,self.layer_num)
            self.layer_id += [23] # 加入层id列表中
            self.layer_num += 1
            channels = channels * len(linaer_layer_output_index_list)
      
      return self.layer_num - 1,channels #返回块输出元素
    
    def make_linear_layers_3d(self,last_block_index,input_size,final_output_size):
      '''
      生成一个线性的，即没有分支的CNN块
      例如：
      nn.Conv2d(1, 6, 3, padding=1),
      nn.BatchNorm2d(6),或者dropout
      nn.ReLU(True),
      nn.MaxPool2d(2, 2)
      '''
      # print("生成make_linear_layers_2d...",input_size,final_output_size)
      no_conv_prob = 0.1
      more_conv_prob = 0.2
      max_conv_num = 3
      last_layer_input_size = input_size
      if(random.randint(1,100) > no_conv_prob*100 or input_size[1] != final_output_size[1]):
        #有conv层，如果没有，以pool层替换，并省略后续的Relu等
        #前提条件是channels数相等，否则无法取消conv层
        conv_num = 1
        while random.randint(1,100) <= 20:
          #计算叠加几个卷积层
          conv_num += 1
          if(conv_num == max_conv_num):
            break
        # print(conv_num)
        #加入CNN层
        for i in range(0,conv_num):
          ConvTranspose = False
          if(i==conv_num-1):
            #如果是最后一个层了，要把channel数一致
            params = self.make_layer(2,last_layer_input_size,final_output_size)
          else:
            if random.randint(1,100) <= 20:
              #卷积层
              params = self.make_layer(2,last_layer_input_size)
            else:
              #反卷积层
              params = self.make_layer(5,last_layer_input_size)
              ConvTranspose = True
          output_size = params[5:10]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          if(i==0):
            #表示接收上一块的合并节点的输入
            link_list=[last_block_index]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          else:
            #表示接收上个节点作为输入
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          if ConvTranspose:
            self.layer_id += [5] # 加入层id列表中
            self.layer_num += 1
          else:

            self.layer_id += [2] # 加入层id列表中
            self.layer_num += 1
          ConvTranspose = False

        #加入BatchNorm或dropout层
        if(random.randint(1,100) < self.batchNorm_prob*100):
          #加入BatchNorm
          params = self.make_layer(17,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [17] # 加入层id列表中
          self.layer_num += 1
        else:
          if(random.randint(1,100) < self.dropout_prob*100):
            #加入Dropout
            dropout_type = self.prob_random([20,26],[0.8,0.2])
            params = self.make_layer(dropout_type,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [dropout_type] # 加入层id列表中
            self.layer_num += 1
        
        #加入激活层
        params = self.make_layer(22,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1

        #加入pool、unpool层

      #加入Pool层
      if random.randint(1,100) >= 20:
        params = self.make_layer(8,last_layer_input_size,final_output_size)
        output_size = params[5:10]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [8] # 加入层id列表中
        self.layer_num += 1
        pool_type = params[-1]
        if pool_type == 0 and random.randint(1,100) <= 20:
          params = self.make_layer(11,last_layer_input_size,final_output_size)
          output_size = params[5:10]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [11] # 加入层id列表中
          self.layer_num += 1        
      else:
        params = self.make_layer(14,last_layer_input_size,final_output_size)
        output_size = params[5:10]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [14] # 加入层id列表中
        self.layer_num += 1        
      return self.layer_num - 1 #返回该线性序列的最后一个节点的索引号

    def make_linear_layers_2d(self,last_block_index,input_size,final_output_size):
      '''
      生成一个线性的，即没有分支的CNN块
      例如：
      nn.Conv2d(1, 6, 3, padding=1),
      nn.BatchNorm2d(6),或者dropout
      nn.ReLU(True),
      nn.MaxPool2d(2, 2)
      '''
      # print("生成make_linear_layers_2d...",input_size,final_output_size)
      no_conv_prob = 0.05
      more_conv_prob = 0.2
      max_conv_num = 3
      last_layer_input_size = input_size
      if(random.randint(1,100) > no_conv_prob*100 or input_size[1] != final_output_size[1]):
        #有conv层，如果没有，以pool层替换，并省略后续的Relu等
        #前提条件是channels数相等，否则无法取消conv层
        conv_num = 2
        while random.randint(1,100) <= 40:
          #计算叠加几个卷积层
          conv_num += 1
          if(conv_num == max_conv_num):
            break
        # print(conv_num)
        #加入CNN层
        for i in range(0,conv_num):
          ConvTranspose = False
          if(i==conv_num-1):
            #如果是最后一个层了，要把channel数一致
            params = self.make_layer(1,last_layer_input_size,final_output_size)
          else:
            if random.randint(1,100) <= 97:
              #卷积层
              params = self.make_layer(1,last_layer_input_size)
            else:
              #反卷积层
              params = self.make_layer(4,last_layer_input_size)
              ConvTranspose = True
          output_size = params[4:8]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          if(i==0):
            #表示接收上一块的合并节点的输入
            link_list=[last_block_index]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          else:
            #表示接收上个节点作为输入
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          if ConvTranspose:
            self.layer_id += [4] # 加入层id列表中
            self.layer_num += 1
          else:
            self.layer_id += [1] # 加入层id列表中
            self.layer_num += 1
          ConvTranspose = False
          #加入激活层
          params = self.make_layer(22,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          if(random.randint(1,100) < 80):
            #加入BatchNorm
            params = self.make_layer(16,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [16] # 加入层id列表中
            self.layer_num += 1
          

          if(random.randint(1,100) < 30 and not self.no_dropout):
            #加入Dropout
            dropout_type = self.prob_random([19,25],[0.8,0.2])
            params = self.make_layer(dropout_type,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [dropout_type] # 加入层id列表中
            self.layer_num += 1

      #加入Pool层
      if random.randint(1,100) >= 50:
        params = self.make_layer(7,last_layer_input_size,final_output_size)
        output_size = params[4:8]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [7] # 加入层id列表中
        self.layer_num += 1
        pool_type = params[-1]
        if pool_type == 0 and random.randint(1,100) <= 20:
          params = self.make_layer(10,last_layer_input_size,final_output_size)
          output_size = params[4:8]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [10] # 加入层id列表中
          self.layer_num += 1        
      else:
        params = self.make_layer(13,last_layer_input_size,final_output_size)
        output_size = params[4:8]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [13] # 加入层id列表中
        self.layer_num += 1        
      return self.layer_num - 1 #返回该线性序列的最后一个节点的索引号

    def make_Bottleneck_layers_2d(self,last_block_index,input_size,final_output_size):
      '''
      生成一个线性的，即没有分支的CNN块
      例如：
      nn.Conv2d(1, 6, 3, padding=1),
      nn.BatchNorm2d(6),或者dropout
      nn.ReLU(True),
      nn.MaxPool2d(2, 2)
      '''
      # print("make_Bottleneck_layers_2d...",input_size,final_output_size)
      more_conv_prob = 0.2
      max_conv_num = 5
      conv_num = 3
      last_layer_input_size = input_size
      if(input_size[1] != final_output_size[1]):
        #有conv层，如果没有，以pool层替换，并省略后续的Relu等
        #前提条件是channels数相等，否则无法取消conv层
        conv_num = 3
        while random.randint(1,100) <= 20:
          #计算叠加几个卷积层
          conv_num += 1
          if(conv_num == max_conv_num):
            break
        mid_channels = self.prob_random([int(input_size[1]/6),int(input_size[1]/4),int(input_size[1]/2)],[0.3,0.4,0.3])
        # print(conv_num)
        #加入CNN层
        for i in range(0,conv_num):
          if(i==conv_num-1):
            #如果是最后一个层了，要把channel数一致
            params = self.make_layer(1,last_layer_input_size,final_output_size)
          else:
            params = self.make_layer(1,last_layer_input_size,[last_layer_input_size[0],mid_channels,last_layer_input_size[2],last_layer_input_size[3]])
          output_size = params[4:8]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          if(i==0):
            #表示接收上一块的合并节点的输入
            link_list=[last_block_index]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          else:
            #表示接收上个节点作为输入
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            
          self.layer_id += [1] # 加入层id列表中
          self.layer_num += 1
          #加入激活层
          params = self.make_layer(22,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          ConvTranspose = False
          if(random.randint(1,100) < 80):
            #加入BatchNorm
            params = self.make_layer(16,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [16] # 加入层id列表中
            self.layer_num += 1
          

          if(random.randint(1,100) < 30 and not self.no_dropout):
            #加入Dropout
            dropout_type = self.prob_random([19,25],[0.8,0.2])
            params = self.make_layer(dropout_type,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [dropout_type] # 加入层id列表中
            self.layer_num += 1
      

        #加入pool、unpool层

      #加入Pool层
      if random.randint(1,100) >= 80:
        params = self.make_layer(7,last_layer_input_size,final_output_size)
        output_size = params[4:8]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [7] # 加入层id列表中
        self.layer_num += 1
        pool_type = params[-1]
        if pool_type == 0 and random.randint(1,100) <= 10:
          params = self.make_layer(10,last_layer_input_size,final_output_size)
          output_size = params[4:8]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [10] # 加入层id列表中
          self.layer_num += 1        
      else:
        params = self.make_layer(13,last_layer_input_size,final_output_size)
        output_size = params[4:8]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [13] # 加入层id列表中
        self.layer_num += 1        
      return self.layer_num - 1 #返回该线性序列的最后一个节点的索引号

    def make_linear_layers_1d(self,last_block_index,input_size,final_output_size):
      '''
      生成一个线性的，即没有分支的CNN块
      例如：
      nn.Conv2d(1, 6, 3, padding=1),
      nn.BatchNorm2d(6),或者dropout
      nn.ReLU(True),
      nn.MaxPool2d(2, 2)
      '''
      # print("生成make_linear_layers_2d...",input_size,final_output_size)
      no_conv_prob = 0.1
      more_conv_prob = 0.2
      max_conv_num = 3
      last_layer_input_size = input_size
      if(random.randint(1,100) > no_conv_prob*100 or input_size[1] != final_output_size[1]):
        #有conv层，如果没有，以pool层替换，并省略后续的Relu等
        #前提条件是channels数相等，否则无法取消conv层
        conv_num = 1
        while random.randint(1,100) <= 20:
          #计算叠加几个卷积层
          conv_num += 1
          if(conv_num == max_conv_num):
            break
        # print(conv_num)
        #加入CNN层
        for i in range(0,conv_num):
          ConvTranspose = False
          if(i==conv_num-1):
            #如果是最后一个层了，要把channel数一致
            params = self.make_layer(0,last_layer_input_size,final_output_size)
          else:
            if random.randint(1,100) <= 20:
              #卷积层
              params = self.make_layer(0,last_layer_input_size)
            else:
              #反卷积层
              params = self.make_layer(3,last_layer_input_size)
              ConvTranspose = True
          output_size = params[3:6]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          if(i==0):
            #表示接收上一块的合并节点的输入
            link_list=[last_block_index]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          else:
            #表示接收上个节点作为输入
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
          if ConvTranspose:
            self.layer_id += [3] # 加入层id列表中
            self.layer_num += 1
          else:

            self.layer_id += [0] # 加入层id列表中
            self.layer_num += 1
          ConvTranspose = False

        #加入BatchNorm或dropout层
        if(random.randint(1,100) < self.batchNorm_prob*100):
          #加入BatchNorm
          params = self.make_layer(15,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [15] # 加入层id列表中
          self.layer_num += 1
        else:
          if(random.randint(1,100) < self.dropout_prob*100):
            #加入Dropout
            dropout_type = 18
            params = self.make_layer(dropout_type,last_layer_input_size)
            self.layer_parameters += params #加入参数向量中
            link_list=[self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [dropout_type] # 加入层id列表中
            self.layer_num += 1
        
        #加入激活层
        params = self.make_layer(22,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1

        #加入pool、unpool层

      #加入Pool层
      if random.randint(1,100) >= 20:
        params = self.make_layer(6,last_layer_input_size,final_output_size)
        output_size = params[3:6]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [6] # 加入层id列表中
        self.layer_num += 1
        pool_type = params[-1]
        if pool_type == 0 and random.randint(1,100) <= 20:
          params = self.make_layer(9,last_layer_input_size,final_output_size)
          output_size = params[3:6]
          last_layer_input_size = output_size
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [9] # 加入层id列表中
          self.layer_num += 1
      else:
        params = self.make_layer(12,last_layer_input_size,final_output_size)
        output_size = params[3:6]
        last_layer_input_size = output_size
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [12] # 加入层id列表中
        self.layer_num += 1

      return self.layer_num - 1 #返回该线性序列的最后一个节点的索引号
      
    def make_layer(self,layer_id,input_size,output_size=None,add_num=None,out_channels=None):
        # print(layer_id,input_size,output_size)
        '''
        layer_id:神经网络层的id
        input_size: list
        返回参数向量
        '''
        # print(layer_id,input_size,output_size)
        if(layer_id == 0):
          # print("make conv2d...")
          #现在写的是kernel_size是正方形或者立方体
          input_length = input_size[2]
          in_channels,out_channels,kernel_size,stride_size,padding_size,dilation_size,groups = [0 for index in range(7)]
          in_channels = input_size[1]
          # print(in_channels,'in_channels start')
          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(32,64)
            else:
              out_channels = random.randint(int(in_channels*2),in_channels*3)
            
            output_size = [0,0,0]
            output_size = [input_size[0],out_channels,input_length]
          else:
            out_channels = output_size[1]
          #生成宽高一样的kernel
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
          #生成stride,padding
          if(output_size==None):
            stride_size = 1
            padding_size = random.randint(0,kernel_size)
            #计算output_size
            out_length = (input_length + 2*padding_size - dilation_size*(kernel_size-1) - 1)/(stride_size) + 1
            output_size = [input_size[0],out_channels,out_length]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_length = output_size[2]
            in_length = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
              assert find_count < 200, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                stride_size = (in_length + 2*p - dilation_size*(kernel_size-1) - 1)/(out_length - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
          # print(in_channels,'in_channels')
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size,stride_size,padding_size,dilation_size,groups]]

        elif(layer_id == 1):
          # print('1conv2d',input_size,output_size)
          low_channel_prob = 0.5
          # print("make conv2d...")
          #现在写的是kernel_size是正方形或者立方体
          input_height = input_size[2]
          input_width = input_size[3]

          in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups = [0 for index in range(11)]
          in_channels = input_size[1]
          # print(in_channels,'in_channels start')
          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(32,64)
            else:
              # print(int(in_channels/0.8),int(in_channels*1.2))
              out_channels = random.randint(int(in_channels*0.8),int(in_channels*1.2))
              if self.large == 1 and out_channels > 600:
                out_channels = self.prob_random([int(out_channels/6),int(out_channels/5),int(out_channels/4)],[0.3,0.5,0.2])
              elif self.large == 0 and out_channels > 200:
                out_channels = self.prob_random([int(out_channels/6),int(out_channels/5),int(out_channels/4)],[0.3,0.5,0.2])
                
            
            output_size = [0,0,0,0]
            output_size = [input_size[0],out_channels,input_height,input_width]
          else:
            out_channels = output_size[1]
          #生成宽高一样的kernel
          # print('2conv2d',input_size,output_size)
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.78,0.05,0.1,0.04,0.01,0.01,0.01])
          kernel_size_height,kernel_size_width = kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          dilation_height,dilation_width = dilation_size,dilation_size
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
          #生成stride,padding
          if(output_size==None):
            stride_size = 1
            stride_height,stride_width = stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_height,padding_width = padding_size,padding_size
            #计算output_size
            out_height = math.floor((input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1)
            out_width = math.floor((input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1)
            output_size = [input_size[0],out_channels,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_height = output_size[2]
            in_height = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
              assert find_count < 200, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                stride_size = (in_height + 2*p - dilation_size*(kernel_size-1) - 1)/(out_height - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size
          # print(in_channels,'in_channels')
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups]]
          
        elif(layer_id == 2):
          # print("make conv2d...")
          #现在写的是kernel_size是正方形或者立方体
          input_depth = input_size[2]
          input_height = input_size[3]
          input_width = input_size[4]

          in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,\
          stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_width,dilation_height,\
          groups = [0 for index in range(15)]

          in_channels = input_size[1]
          # print(in_channels,'in_channels start')
          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(32,64)
            else:
              out_channels = random.randint(int(in_channels*2),in_channels*3)
            
            output_size = [0,0,0,0,0]
            output_size = [input_size[0],out_channels,input_depth,input_height,input_width]
          else:
            out_channels = output_size[1]
          #生成宽高一样的kernel
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_depth,kernel_size_height,kernel_size_width = kernel_size,kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          dilation_depth,dilation_height,dilation_width = dilation_size,dilation_size,dilation_size
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
          # print('in_channels,out_channels,common_divisor',in_channels,out_channels,common_divisor)
          #生成stride,padding
          if(output_size==None):
            stride_size = 1
            stride_depth,stride_height,stride_width = stride_size,stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_depth,padding_height,padding_width = padding_size,padding_size,padding_size
            #计算output_size
            out_depth = (input_depth + 2*padding_depth - dilation_depth*(kernel_size_depth-1) - 1)/(stride_depth) + 1
            out_height = (input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1
            out_width = (input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1
            output_size = [input_size[0],out_channels,out_depth,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            #先寻找height width
            out_height = output_size[3]
            in_height = input_size[3]
            find = False
            find_count = 0
            while not find:
              find_count += 1
              assert find_count < 200, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                stride_size = (in_height + 2*p - dilation_size*(kernel_size-1) - 1)/(out_height - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size

            out_depth = output_size[2]
            in_depth = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
              assert find_count < 200, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                stride_size = (in_depth + 2*p - dilation_size*(kernel_size-1) - 1)/(out_depth - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_depth = padding_size
                  stride_depth = stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_depth = kernel_size
            
          # print(in_channels,'in_channels')
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_width,dilation_height,groups]]          
        
        elif(layer_id == 3):
          # print(output_size,'output_size')
          assert output_size==None,"反卷积层只支持不指定输出大小"
       
          input_length = input_size[2]
          in_channels,out_channels,kernel_size,stride,padding,output_padding,dilation,groups = [0 for index in range(8)]
          in_channels = input_size[1]  

          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(16,64)
            else:
              out_channels = random.randint(int(in_channels*2),in_channels*3)
            
            output_size = [0,0,0,0]
            output_size = [input_size[0],out_channels,input_length]

          #生成宽高一样的kernel
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          output_padding_size = 0
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
            #生成stride,padding
          if(True):
            stride_size = 1
            padding_size = random.randint(0,kernel_size)
            ouput_length = output_padding_size + stride_size*(input_length - 1) - 2*padding_size + dilation_size*(kernel_size - 1) + 1
            #计算output_size
            output_size = [input_size[0],out_channels,ouput_length]
          # print([int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size,stride,padding,output_padding,dilation,groups]])
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size,stride_size,padding,output_padding,dilation_size,groups]]

        elif(layer_id == 4):

          #由于size无法成倍缩小，暂时只支持output_size = None

          # print(output_size,'output_size')
          assert output_size==None,"反卷积层只支持不指定输出大小"
       
          input_height = input_size[2]
          input_width = input_size[3]
          in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,\
          stride_width,padding_height,padding_width,dilation,groups,output_padding_height,output_padding_width = [0 for index in range(12)]
          in_channels = input_size[1]  

          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(16,64)
            else:
              out_channels = random.randint(int(in_channels*2),in_channels*3)
            
            output_size = [0,0,0,0]
            output_size = [input_size[0],out_channels,input_height,input_width]

          #生成宽高一样的kernel
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_height,kernel_size_width = kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          output_padding_height = 0
          output_padding_width = 0
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
            #生成stride,padding
          if(True):
            stride_size = 1
            stride_height,stride_width = stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_height,padding_width = padding_size,padding_size
            ouput_height = output_padding_height + stride_height*(input_height - 1) - 2*padding_height + dilation_size*(kernel_size_height - 1) + 1
            ouput_width = output_padding_width + stride_width*(input_width - 1) - 2*padding_width + dilation_size*(kernel_size_width - 1) + 1
            # output_padding_height = ouput_height - stride_height*(input_height - 1) + 2*padding_height - dilation_size*(kernel_size_height - 1) - 1
            # output_padding_width = ouput_width - stride_width*(input_width - 1) + 2*padding_width - dilation_size*(kernel_size_width - 1) - 1
            #计算output_size
            output_size = [input_size[0],out_channels,ouput_height,ouput_width]
          # print([int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,output_padding_height,output_padding_width,dilation_size,groups]])
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,output_padding_height,output_padding_width,dilation_size,groups]]
          
          # return nn.ConvTranspose2d(in_channels,out_channels,(kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
          #         padding=(padding_height, padding_width),output_padding=(output_padding_height,output_padding_width),dilation=dilation,groups=groups)

        elif(layer_id == 5):

          # print(output_size,'output_size')
          assert output_size==None,"反卷积层只支持不指定输出大小"
          input_depth = input_size[2]
          input_height = input_size[3]
          input_width = input_size[4]
          in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,\
          stride_height,stride_width,padding_depth,padding_height,padding_width,output_padding_depth,output_padding_height,output_padding_width,\
          dilation,groups = [0 for index in range(16)]
          in_channels = input_size[1]  

          if(output_size==None):
            
            if(in_channels <= 3):
              out_channels = random.randint(16,64)
            else:
              out_channels = random.randint(int(in_channels*2),in_channels*3)
            
            output_size = [0,0,0,0]
            output_size = [input_size[0],out_channels,input_depth,input_height,input_width]

          #生成宽高一样的kernel
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_depth,kernel_size_height,kernel_size_width = kernel_size,kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          common_divisor = self.get_common_divisor(in_channels,out_channels)
          output_padding_height = 0
          output_padding_width = 0
          output_padding_depth = 0
          if(len(common_divisor) == 1):
            #只有公约数1
            groups = 1
          else:
            groups = self.prob_random(common_divisor,[0.95]+[(1-0.95)/(len(common_divisor)-1) for i in range(len(common_divisor)-1)])
            #生成stride,padding
          if(True):
            stride_size = 1
            stride_height,stride_width,stride_depth = stride_size,stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_depth,padding_height,padding_width = padding_size,padding_size,padding_size
            ouput_depth = output_padding_depth + stride_depth*(input_depth - 1) - 2*padding_depth + dilation_size*(kernel_size_depth - 1) + 1
            ouput_height = output_padding_height + stride_height*(input_height - 1) - 2*padding_height + dilation_size*(kernel_size_height - 1) + 1
            ouput_width = output_padding_width + stride_width*(input_width - 1) - 2*padding_width + dilation_size*(kernel_size_width - 1) + 1
            # output_padding_height = ouput_height - stride_height*(input_height - 1) + 2*padding_height - dilation_size*(kernel_size_height - 1) - 1
            # output_padding_width = ouput_width - stride_width*(input_width - 1) + 2*padding_width - dilation_size*(kernel_size_width - 1) - 1
            #计算output_size
            output_size = [input_size[0],out_channels,ouput_depth,ouput_height,ouput_width]
          # print([int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,\
          # stride_height,stride_width,padding_depth,padding_height,padding_width,output_padding_depth,output_padding_height,output_padding_width,\
          # dilation_size,groups]])
          return [int(i) for i in input_size+output_size+[in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,\
          stride_height,stride_width,padding_depth,padding_height,padding_width,output_padding_depth,output_padding_height,output_padding_width,\
          dilation_size,groups]]
        
          # return nn.ConvTranspose3d(in_channels,out_channels,(kernel_size_depth, kernel_size_height, kernel_size_width), stride=(stride_depth, stride_height, stride_width),\
          #         padding=(padding_depth, padding_height, padding_width),output_padding=(output_padding_depth, output_padding_height, output_padding_width),dilation=dilation,groups=groups)
        
        elif(layer_id == 6):
          #如果是max pooling则需要返回indices
          input_length = input_size[2]
          input_channels = input_size[1]
          kernel_size,stride_size,padding_size,dilation_size,pool_type = [0 for index in range(5)]
          pool_type = random.randint(0,1)
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          if(pool_type == 1):
            dilation_size = 1

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            padding_size = random.randint(0,kernel_size)
            #计算output_size
            out_length = (input_length + 2*padding_size - dilation_size*(kernel_size-1) - 1)/(stride_size) + 1
            output_size = [input_size[0],input_channels,out_length]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_length = output_size[2]
            input_length = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                
                stride_size = (input_length + 2*p - dilation_size*(kernel_size-1) - 1)/(out_length - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])

          return [int(i) for i in input_size+output_size+[kernel_size,stride_size,padding_size,dilation_size,pool_type]]
        
        elif(layer_id == 7):
          #如果是max pooling则需要返回indices
          input_height = input_size[2]
          input_width = input_size[3]
          input_channels = input_size[1]
          kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,pool_type = [0 for index in range(9)]
          pool_type = random.randint(0,1)
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_height,kernel_size_width = kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          dilation_height,dilation_width = dilation_size,dilation_size
          if(pool_type == 1):
            dilation_height,dilation_width = 1,1

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            stride_height,stride_width = stride_size,stride_size
            padding_size = 0
            padding_height,padding_width = padding_size,padding_size
            #计算output_size
            out_height = math.floor((input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1)
            out_width = math.floor((input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1)
            output_size = [input_size[0],input_channels,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_height = output_size[2]
            in_height = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                
                stride_size = (in_height + 2*p - dilation_size*(kernel_size-1) - 1)/(out_height - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size

          return [int(i) for i in input_size+output_size+[kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,pool_type]]

        elif(layer_id == 8):
          #如果是max pooling则需要返回indices
          input_depth = input_size[2]
          input_height = input_size[3]
          input_width = input_size[4]
          input_channels = input_size[1]
          kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,\
          stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_height,dilation_width,pool_type = [0 for index in range(13)]
          pool_type = random.randint(0,1)
          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_depth,kernel_size_height,kernel_size_width = kernel_size,kernel_size,kernel_size
          #生成宽高一样的dilation
          dilation_size = self.prob_random([1,2],[0.95,0.05])
          dilation_depth,dilation_height,dilation_width = dilation_size,dilation_size,dilation_size
          if(pool_type == 1):
            dilation_depth,dilation_height,dilation_width = 1,1,1

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            stride_depth,stride_height,stride_width = stride_size,stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_depth,padding_height,padding_width = padding_size,padding_size,padding_size
            #计算output_size
            out_depth = (input_depth + 2*padding_depth - dilation_depth*(kernel_size_depth-1) - 1)/(stride_depth) + 1
            out_height = (input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1
            out_width = (input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1
            output_size = [input_size[0],input_channels,out_depth,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_height = output_size[3]
            in_height = input_size[3]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                
                stride_size = (in_height + 2*p - dilation_size*(kernel_size-1) - 1)/(out_height - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size
            
            out_depth = output_size[2]
            in_depth = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for p in range(0,int(kernel_size/2)+1):
                
                stride_size = (in_depth + 2*p - dilation_size*(kernel_size-1) - 1)/(out_depth - 1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = p
                  padding_depth = padding_size
                  stride_depth = stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_depth = kernel_size

          return [int(i) for i in input_size+output_size+[kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,\
          stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_height,dilation_width,pool_type]]

        elif(layer_id == 9):

          input_channels = input_size[1]
          input_length = input_size[2]
          kernel_size,stride_size,padding_size = [0 for index in range(3)]

          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            padding_size = random.randint(0,kernel_size)
            #计算output_size
            out_length = (input_length-1)*stride_size - 2*padding_size + kernel_size
            output_size = [input_size[0],input_channels,out_length]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_length = output_size[2]
            input_length = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for padding in range(0,int(kernel_size/2)+1):
                stride_size = (2*padding - kernel_size + out_length) / (input_length-1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = padding
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])

          return [int(i) for i in input_size+output_size+[kernel_size,stride_size,padding_size]]

          # return nn.MaxUnpool2d(kernel_size = (kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),padding=(padding_height, padding_width))
        
          # return nn.MaxUnpool1d(kernel_size = kernel_size, stride=stride, padding=padding)
        
        elif(layer_id == 10):

          input_channels = input_size[1]
          input_height = input_size[2]
          input_width = input_size[3]
          kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width = [0 for index in range(6)]

          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_height,kernel_size_width = kernel_size,kernel_size

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            stride_height,stride_width = stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_height,padding_width = padding_size,padding_size
            #计算output_size
            out_height = (input_height-1)*stride_height - 2*padding_height + kernel_size_height
            out_width = (input_width-1)*stride_width - 2*padding_width + kernel_size_width
            output_size = [input_size[0],input_channels,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_height = output_size[2]
            in_height = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for padding in range(0,int(kernel_size/2)+1):
                stride_size = (2*padding - kernel_size_height + out_height) / (input_height-1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = padding
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size

          return [int(i) for i in input_size+output_size+[kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width]]

          # return nn.MaxUnpool2d(kernel_size = (kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),padding=(padding_height, padding_width))
        
        elif(layer_id == 11):

          input_channels = input_size[1]
          input_depth = input_size[2]
          input_height = input_size[3]
          input_width = input_size[4]
          kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width = [0 for index in range(9)]

          kernel_size = self.prob_random([1,2,3,4,5,6,7],[0.05,0.05,0.4,0.05,0.3,0.05,0.1])
          kernel_size_depth,kernel_size_height,kernel_size_width = kernel_size,kernel_size,kernel_size

          if(output_size==None):
            # stride_size = self.prob_random([1,2,3],[0.6,0.3,0.1])
            stride_size = 1
            stride_depth,stride_height,stride_width = stride_size,stride_size,stride_size
            padding_size = random.randint(0,kernel_size)
            padding_depth,padding_height,padding_width = padding_size,padding_size,padding_size
            #计算output_size
            out_depth = (input_depth-1)*stride_depth - 2*padding_depth + kernel_size_depth
            out_height = (input_height-1)*stride_height - 2*padding_height + kernel_size_height
            out_width = (input_width-1)*stride_width - 2*padding_width + kernel_size_width
            output_size = [input_size[0],input_channels,out_depth,out_height,out_width]
          else:
            #通过已知的kernel_size,dilation_size计算stride_size,padding_size的整数解
            out_height = output_size[3]
            in_height = input_size[3]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for padding in range(0,int(kernel_size/2)+1):
                stride_size = (2*padding - kernel_size_height + out_height) / (input_height-1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = padding
                  padding_height,padding_width = padding_size,padding_size
                  stride_height,stride_width = stride_size,stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_height,kernel_size_width = kernel_size,kernel_size
                
            out_depth = output_size[2]
            in_depth = input_size[2]
            find = False
            find_count = 0
            while not find:
              find_count += 1
                # print(kernel_size)
              assert find_count < 30, "疑似找不到符合要求的神经网络层"
              for padding in range(0,int(kernel_size/2)+1):
                stride_size = (2*padding - kernel_size_height + out_height) / (input_height-1)
                if(stride_size.is_integer() and stride_size > 0):
                  padding_size = padding
                  padding_depth = padding_size
                  stride_depth = stride_size
                  find = True
                  break
              else:
                kernel_size = self.prob_random([1,2,3,4,5,6,7],[1/7 for i in range(7)])
                kernel_size_depth = kernel_size

          return [int(i) for i in input_size+output_size+[kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width]]
        
        elif(layer_id == 12):
          pool_type = random.randint(0,1)
          if(output_size==None):
            return input_size + input_size + [pool_type]
          else:
            return input_size + output_size + [pool_type]
        
        elif(layer_id == 13):
          pool_type = random.randint(0,1)
          if(output_size==None):
            return input_size + input_size + [pool_type]
          else:
            return input_size + output_size + [pool_type]
        
        elif(layer_id == 14):
          pool_type = random.randint(0,1)
          if(output_size==None):
            return input_size + input_size + [pool_type]
          else:
            return input_size + output_size + [pool_type]
        
        elif(layer_id == 15):
          num_features = input_size[1]
          return input_size + [num_features]

        elif(layer_id == 16):
          num_features = input_size[1]
          return input_size + [num_features]

        elif(layer_id == 17):
          num_features = input_size[1]
          return input_size + [num_features]
        
        elif(layer_id == 18):
          probability = self.prob_random([0.1,0.2,0.3,0.4,0.5],[0.2 for i in range(5)])
          return input_size+[probability]

        elif(layer_id == 19):
          probability = self.prob_random([0.1,0.2,0.3,0.4,0.5],[0.2 for i in range(5)])
          return input_size+[probability]
          
        elif(layer_id == 20):
          probability = self.prob_random([0.1,0.2,0.3,0.4,0.5],[0.2 for i in range(5)])
          return input_size+[probability]
          
        elif(layer_id == 21):
          return input_size+output_size
          
        elif(layer_id == 22):
          activation_type = random.randint(1,4)
          length = input_size[1]
          for i in range(2,len(input_size)):
            length *= input_size[i]
          if(activation_type == 1):
            return [input_size[0],length] + [1,0,0,0]
          elif(activation_type == 2):
            return [input_size[0],length] + [0,1,0,0]
          elif(activation_type == 3):
            return [input_size[0],length] + [0,0,1,0]
          else:
            return [input_size[0],length] + [0,0,0,1]
        
        #由于add和concat不是实际的神经网络层，随便返回一个神经网络层
        elif(layer_id == 23):
          length = input_size[1]
          for i in range(2,len(input_size)):
            length *= input_size[i]
          return [input_size[0],length]+[out_channels]

        elif(layer_id == 24):
          length = input_size[1]
          for i in range(2,len(input_size)):
            length *= input_size[i]
          return [input_size[0],length]+[add_num]

        elif(layer_id == 25):
          probability = self.prob_random([0.1,0.2,0.3,0.4,0.5],[0.2 for i in range(5)])
          return input_size+[probability]

        elif(layer_id == 26):
          probability = self.prob_random([0.1,0.2,0.3,0.4,0.5],[0.2 for i in range(5)])
          return input_size+[probability]

    def get_net_input_size(self):
      if(self.dimension == 1):
        return [1,1,random.randint(100,10000)]
      elif(self.dimension == 2):
        pic_edge_length = random.randint(28,224)
        if(random.randint(1,100) < 15):
          pic_edge_length = 224
        pic_edge_length = 224
        return [1,random.randint(1,3),pic_edge_length,pic_edge_length]
      else:
        pic_edge_length = random.randint(28,112)
        video_frames = random.randint(15,80)
        return [1,random.randint(1,3),video_frames,pic_edge_length,pic_edge_length]

    def get_layer_output_size(self,params,input_size):
      '''
      根据一层的输入尺寸，得到该层的输出尺寸
      params:list
      input_size:list
      return list
      '''
      
    def prob_random(self,arr1,arr2):
      '''
      指定概率，获取随机数
      '''
      assert len(arr1) == len(arr2), "Length does not match."
      # assert sum(arr2) == 1 , "Total rate is not 1."

      sup_list = [len(str(i).split(".")[-1]) for i in arr2]
      top = 10 ** max(sup_list)
      new_rate = [int(i*top) for i in arr2]
      rate_arr = []
      for i in range(1,len(new_rate)+1):
        rate_arr.append(sum(new_rate[:i]))
      rand = random.randint(1,top)
      data = None
      for i in range(len(rate_arr)):
        if rand <= rate_arr[i]:
          data = arr1[i]
          break
      return data

    def get_common_divisor(self,a,b):
      '''
      获得两个数的所有公约数
      '''
      common_divisor_list = [1]
      for i in range(2,max(a,b)):
        if(a%i == 0 and b%i == 0):
          common_divisor_list.append(i)
      return common_divisor_list

    def get_link_vector(self,link_list,target_layer_index):
      '''
      生成一个节点的连接向量
      target_layer_index表示该节点在layer_id中的数组下标
      '''
      link_vector = [0 for i in range(target_layer_index+1)]
      for i in range(0,len(link_list)):
        if(link_list[i] == -1):
          #表示接收初始输入
          link_vector[target_layer_index] = 1
        else:
          link_vector[link_list[i]] = 1
      return link_vector

    def get_params_length(self,layer_id):
      '''
      获取不同层参数向量长度
      '''
      get_params_length_dic = {
        0:13,
        1:19,
        2:25,
        3:14,
        4:20,
        5:26,
        6:11,
        7:17,
        8:23,
        9:9,
        10:14,
        11:19,
        12:7,
        13:9,
        14:11,
        15:4,
        16:5,
        17:6,
        18:4,
        19:5,
        20:6,
        21:4,
        22:6,
        23:3,
        24:3,
        25:5,
        26:6,
      }
      return get_params_length_dic[layer_id]
 
    def get_params_num(self,layer_id,params_list):
      '''
      计算一个层的参数数量
      '''
      # print(layer_id,params_list)
      if layer_id == 0:

        input_channels,output_channels,kernel_height,kernel_width = params_list[1],params_list[5],params_list[10],params_list[11]
        return input_channels*kernel_height*kernel_width*output_channels

      elif layer_id == 1:

        input_channels,output_channels,kernel_length = params_list[1],params_list[4],params_list[8]
        return input_channels*output_channels*kernel_length

      elif layer_id == 2:
        
        input_channels,output_channels,kernel_size_depth,kernel_size_height,kernel_size_width = params_list[1],params_list[6],params_list[12],params_list[13],params_list[14]
        return input_channels*output_channels*kernel_size_depth*kernel_size_height*kernel_size_width

      elif layer_id == 3:

        input_channels,output_channels,kernel_height,kernel_width = params_list[1],params_list[5],params_list[10],params_list[11]
        return input_channels*kernel_height*kernel_width*output_channels

      elif layer_id == 4:

        input_channels,output_channels,kernel_length = params_list[1],params_list[4],params_list[8]
        return input_channels*output_channels*kernel_length

      elif layer_id == 5:

        input_channels,output_channels,kernel_size_depth,kernel_size_height,kernel_size_width = params_list[1],params_list[6],params_list[12],params_list[13],params_list[14]
        return input_channels*output_channels*kernel_size_depth*kernel_size_height*kernel_size_width

      elif layer_id == 6:

        return 0

      elif layer_id == 7:

        return 0

      elif layer_id == 8:

        return 0

      elif layer_id == 9:

        return 0

      elif layer_id == 10:

        return 0

      elif layer_id == 11:

        return 0

      elif layer_id == 12:

        return 0

      elif layer_id == 13:

        return 0

      elif layer_id == 14:

        return 0

      elif layer_id == 15:

        #不考虑批标准化层的可训练参数数量
        return 0

      elif layer_id == 16:

        #不考虑批标准化层的可训练参数数量
        return 0

      elif layer_id == 17:

        #不考虑批标准化层的可训练参数数量
        return 0

      elif layer_id == 18:

        return 0

      elif layer_id == 19:

        return 0

      elif layer_id == 20:

        return 0

      elif layer_id == 21:

        input_length,output_length = params[1,3]
        return input_length*(output_length+1)

      elif layer_id == 22:

        return 0

      elif layer_id == 23:

        return 0

      elif layer_id == 24:

        return 0

      elif layer_id == 25:

        return 0

      elif layer_id == 26:

        return 0

def prob_random(arr1,arr2):
      '''
      指定概率，获取随机数
      '''
      assert len(arr1) == len(arr2), "Length does not match."
      # assert sum(arr2) == 1 , "Total rate is not 1."

      sup_list = [len(str(i).split(".")[-1]) for i in arr2]
      top = 10 ** max(sup_list)
      new_rate = [int(i*top) for i in arr2]
      rate_arr = []
      for i in range(1,len(new_rate)+1):
        rate_arr.append(sum(new_rate[:i]))
      rand = random.randint(1,top)
      data = None
      for i in range(len(rate_arr)):
        if rand <= rate_arr[i]:
          data = arr1[i]
          break
      return data

def make_net_data(a):
  while(1):
    stream_num = 1
    if random.randint(0,100)<23:
      block_num = random.randint(4,7)
    else:
      block_num = random.randint(8,18)
      
    large = random.randint(0,1)
    try:
      dim = 2
      print(stream_num,block_num,large)
      vg = VectorGenerator(dimension=dim,block_num=block_num,stream_num=stream_num,large = large)
      vg.make_net()
      if not validate_NN(vg,dim):
        stream_num = 1
        if random.randint(0,100)<23:
          block_num = random.randint(4,7)
        else:
          block_num = random.randint(8,18)
        large = random.randint(0,1)
        continue
    except:
      continue
    else:
      break    
  
def get_energy(return_dict,iLocIndexer,net_input): 
  try:
    monitor = Monitor(0.01)
    layer_parameters = iLocIndexer['layer_parameters'].split(',')
    layer_parameters = [float(x) if '.' in x else int(x) for x in layer_parameters]
    layer_link = iLocIndexer['layer_link'].split(',')
    layer_link = [int(x) for x in layer_link]
    layer_id = iLocIndexer['layer_id'].split(',')
    layer_id = [int(x) for x in layer_id]
    NN = NNgenerator(layer_parameters,layer_link,layer_id)
    NN.cuda()
    dim = int(iLocIndexer['dimension'])
    if dim == 1:
        x = torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]))
    elif dim ==2:
        x = torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]),int(net_input[3]))
    elif dim ==3:
        x = torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]),int(net_input[3]),int(net_input[4]))
        
    b_x = Variable(x).cuda()
    
    torch.cuda.synchronize()
    start_time = round(time.time()*1000)
    monitor.begin()
    for i in range(0,10000):
      print(i)
      output = NN(b_x)
      if round(time.time()*1000) - start_time > 15 * 1000 and i >= 5:
          forward_num = i
          print('结束')
          break
      torch.cuda.synchronize()
    monitor.stop()
    time.sleep(2)
    forward_energy = (monitor.forward_energy) / forward_num # mJ
    silence_energy = (monitor.silence) / forward_num # mJ
    all_energy = (monitor.all_energy) / forward_num # mJ
    mean_power = monitor.mean_power
    all_time = monitor.all_time
    monitor.exit()

  except Exception as e:
    print(traceback.print_exc())
    print(repr(e))
    with open(r'energy_%s.txt' % file_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
      file.write("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" + "\n")
      file.close()
  else:
    save_energy(iLocIndexer,forward_energy,silence_energy,all_energy,mean_power,forward_num,all_time)

def save_energy(iLocIndexer,forward_energy,silence_energy,all_energy,mean_power,forward_num,all_time):

    str1 = ''
    str1 += iLocIndexer['layer_parameters'] + " "
    str1 += iLocIndexer['layer_link'] + " "
    str1 += iLocIndexer['layer_id'] + " "
    str1 += str(iLocIndexer['params_num']) + " "
    str1 += str(iLocIndexer['dimension']) + " "
    str1 += str(iLocIndexer['block_num']) + " "
    str1 += str(iLocIndexer['stream_num']) + " "
    str1 += cpu_name + " "
    str1 += cpu_MHz + " "
    str1 += cache_size + " "
    str1 += str(processor_num) + " "
    str1 += gpu_name + " "
    str1 += str(mean_power) + " "
    str1 += str(all_time) + " "
    str1 += str(forward_num) + " "
    str1 += str(forward_energy) + " "
    str1 += str(silence_energy) + " "
    str1 += str(all_energy)
    # print(str1)
    with open(r'energy_%s.txt' % file_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str1 + "\n")
        file.close()
import gc
import sys
if __name__ == '__main__':
    for i in range(0,100000):
        num_processes = 1
        p = mp.Process(target=make_net_data, args=(1,))
        p.start()
        p.join()
        # sys.exit(1)

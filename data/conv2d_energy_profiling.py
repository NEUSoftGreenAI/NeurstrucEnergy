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
import traceback
from Monitor import Monitor
import gc
np.set_printoptions(threshold=500)
file_name = 'PreVIousNet_Conv2d'

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
    monitor = Monitor(0.01)
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
            comp_context[layer_index] = self.layer_list[layer_index](x)
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
              # print(self.layer_list[layer_index])
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
      for i in range(0,length):
        for j in range(0,i+1):
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
  torch.cuda.empty_cache()
  NN = NNgenerator(vg.layer_parameters,vg.layer_link,vg.layer_id)
  NN.cuda()
  if dim == 1:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2])
  elif dim == 2:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2],vg.net_input[3])
  else:
    x = torch.rand(vg.net_input[0],vg.net_input[1],vg.net_input[2],vg.net_input[3],vg.net_input[4])

  b_x = Variable(x).cuda()
  output = NN(b_x)
  total_params = sum(p.numel() for p in NN.parameters())
  # print(f'{total_params:,} total parameters.')
  str1 = ''
  str1 += ",".join('%s' %i for i in vg.layer_parameters) + " "
  str1 += ",".join('%s' %i for i in vg.layer_link) + " "
  str1 += ",".join('%s' %i for i in vg.layer_id) + " "
  str1 += str(total_params) + " "
  str1 += str(vg.dimension) + " "
  str1 += str(vg.block_num) + " "
  str1 += str(vg.stream_num) + " "
  str1 += cpu_name + " "
  str1 += cpu_MHz + " "
  str1 += cache_size + " "
  str1 += str(processor_num) + " "
  str1 += gpu_name + " "
  str1 += "0" + " "
  str1 += "0" + " "
  str1 += "0" + " "
  str1 += "0" + " "
  with open("energy_%s.txt" % file_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
    file.write(str1 + "\n")
    file.close()

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

def get_energy(return_dict,iLocIndexer,params): 
  try:
    monitor = Monitor(0.01)
    layer_parameters = iLocIndexer['data'].split(',')
    layer_parameters = [int(x.replace('.0','')) for x in layer_parameters]
    in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups = layer_parameters[-11:]
    print(in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups)
    conv2d = nn.Conv2d(in_channels,out_channels,(kernel_size_height, kernel_size_width), stride=(stride_height, stride_width),\
                  padding=(padding_height, padding_width),dilation=(dilation_height, dilation_width),groups=groups).cuda()
    net_input = layer_parameters[0:4]
    dim = 2
    if dim == 1:
        x = torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]))
    elif dim ==2:
        x = torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]),int(net_input[3]))
    elif dim ==3:
        x =  torch.rand(int(net_input[0]),int(net_input[1]),int(net_input[2]),int(net_input[3]),int(net_input[4]))
        
    b_x = Variable(x).cuda()
    torch.cuda.synchronize()
    monitor.begin()
    for i in range(0,10000):
      #print(i)
      output = conv2d(b_x)
      torch.cuda.synchronize()
      if i == 15:
          print('begin monier')
          monitor.stopped = False
          start_time = round(time.time()*1000)
      if i > 30 and round(time.time()*1000) - start_time > 4 * 1000:
          forward_num = i - 15
          print('stop total forward num:' + str(i-15))
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
      file.write("0 0 0 0 0 0 0" + "\n")
      file.close()
  else:
    save_energy(iLocIndexer,forward_energy,silence_energy,all_energy,mean_power,forward_num,all_time)

def save_energy(iLocIndexer,forward_energy,silence_energy,all_energy,mean_power,forward_num,all_time):

    str1 = ''
    str1 += iLocIndexer['data'] + " "
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
import os
import serial #导入模块
import time

if __name__ == '__main__':
    time.sleep(20)
    content = os.popen('echo 192 > /sys/devices/pwm-fan/target_pwm').read()
    ser=serial.Serial('/dev/ttyUSB0',9600,timeout=None)
    time.sleep(20)
    hex = b'\x01\x10\x0C\x81\x00\x01\x02\x00\x14\x74\x4E' # 对应就是0xef01ffffffff

    ser.write(hex)

    time.sleep(1)

    hex = b'\x01\x10\x0C\x1C\x00\x01\x02\x00\x06\xE8\x0E' # 对应就是0xef01ffffffff

    ser.write(hex)
    content = os.popen('echo 192 > /sys/devices/pwm-fan/target_pwm')
    df = pd.read_csv(r'energy_%s.txt' % file_name,sep = ' ',index_col=False)
    df_all = pd.read_csv(r'%s.txt' % file_name,sep = ' ',index_col=False)
    target_length = len(df)
    index = target_length
    count = 0
    manager = Manager()
    return_dict = manager.dict()
    model = manager.Queue(1)
    print(index)
    p = mp.Process(target=achieve_stable_energy, args=(return_dict,))
    p.start()
    p.join()
    for i in range(0,100000):
        num_processes = 1
        p = mp.Process(target=get_energy, args=(return_dict,df_all.iloc[index],df_all.iloc[index]['data']))
        p.start()
        p.join()
        count += 1
        if index+1 > len(df_all):
            break
        print("进行",count,"总计",len(df)+count,"index:",index)
        index += 1
        gc.collect()

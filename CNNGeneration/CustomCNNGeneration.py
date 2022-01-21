# -*- coding: utf-8 -*-

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
np.set_printoptions(threshold=500)

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
          #print(self.layer_parameters[self.parameters_flag:self.parameters_flag+params_length], self.layer_id[i],i)
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
              
              #print("父节点",parent_indices[0],comp_context[parent_indices[0]].size())
              #print("当前节点",layer_index)
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
                #print("父节点",parent_indices)
                # for i in range(0,len(parent_indices)):
                  # print(comp_context[parent_indices[i]].size())
                #print("当前节点",layer_index)
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
                    #print('add层',comp_context[layer_index].size(),converge_tuple[i].size())
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
       
       
def validate_NN(vg,dim):
  # torch.cuda.empty_cache()
  NN = NNgenerator(vg.layer_parameters,vg.layer_link,vg.layer_id)
  total_params = sum(p.numel() for p in NN.parameters())
  # if total_params > 500000000 or total_params<600000:
  #   return False
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
  with open("custom_data.txt","a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
    file.write(str1 + "\n")
    file.close()
  return True

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
      ② 对于每个神经网络块，为既定的主流CNN的组成块。
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
    
    def get_outshape_recent_size(self,shape):
      recent_size = 224
      gap = 999
      for size in [112,56,28,14,7]:
        if abs(shape-size) < gap:
          gap = abs(shape-size)
          recent_size = size
      return recent_size
    
    def make_net(self):
      
      if True:
        #生成CNN网络
        for net_stream in range(0,1):
          
          last_block_input_size = self.net_input
          last_block_index = -1
          for block in range(0,self.block_num):
            #[224,112,56,28,14,7,1]
            #[224,56,28,14,7,1]
            #print("netstream",net_stream," 第",block,"个block", '剩余',self.block_num - block - 1)
            #一条中的多个block
            input_batch_size,input_channels,input_height,input_width = last_block_input_size
            out_channels = 1
            out_shape = 0
              # 生成2 3 4整数倍的out channels
            if self.block_num <= 5:
              if input_height == 224:
                out_shape = self.prob_random([56,112],[0.7,0.3])  
              else:
                out_shape = self.prob_random([int(input_height*0.5),input_height],[0.9,0.1])
              recent_size = self.get_outshape_recent_size(out_shape)
              if recent_size == 112 and input_channels > 100:
                out_shape = int(input_height*0.5)
              elif recent_size == 56 and input_channels > 200:
                out_shape = int(input_height*0.5)
              elif recent_size == 28 and input_channels > 500:
                out_shape = int(input_height*0.5) 
              elif recent_size == 14 and input_channels > 1000:
                out_shape = int(input_height*0.5) 
              if out_shape < 7:
                out_shape = input_height
                    
            else:
              if input_height == 224:
                out_shape = self.prob_random([56,112],[0.1,0.9])
              else:
                remain_block = self.block_num - block - 1
                if input_height > 7 * math.pow(2,remain_block):
                  #print('remain_block')
                  out_shape = self.prob_random([int(input_height*0.5),input_height],[0.999,0.001])
                else:
                  #print('random')
                  out_shape = self.prob_random([int(input_height*0.5),input_height],[0.7,0.3])
              if out_shape < 7:
                out_shape = input_height
              
              recent_size = self.get_outshape_recent_size(out_shape)
              if recent_size == 112 and input_channels > 100:
                out_shape = int(input_height*0.5)
              elif recent_size == 56 and input_channels > 200:
                out_shape = int(input_height*0.5)
              elif recent_size == 28 and input_channels > 500:
                out_shape = int(input_height*0.5) 
              elif recent_size == 14 and input_channels > 1000:
                out_shape = int(input_height*0.5) 
            
            # print(input_height,out_shape)
            output_size = [input_batch_size,out_channels,out_shape,out_shape]
            # print('output_size',output_size)
            #block = 0 时，接收初始输入，为last_block_index = -1
            last_block_index,output_size = self.make_block(last_block_input_size,output_size,last_block_index,4)
            last_block_input_size = output_size
            
        fc_length = 0
        input_fc_size = last_block_input_size
        
        #加入dropout
        if random.randint(0,10) < 5:
          params = self.make_layer(19,last_block_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [19] # 加入层id列表中
          self.layer_num += 1
        #生成全连接层 或者全局池化层
        if random.randint(0,10) < 6:
          #生成全局池化层

          #reshape out_channel = 1000
          #加入 #1 conv2d
          in_channel = last_block_input_size[1]
          out_channel = 1000
          conv_params = [in_channel,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
          params = last_block_input_size + [0,0,0,0] + conv_params
          output_size = self.get_output_size(params,1,last_block_input_size)
          params = last_block_input_size + output_size + conv_params
          self.layer_parameters += params #加入参数向量中
          link_list=[last_block_index]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [1] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          
          #加入 #2 ReLU
          length = last_layer_input_size[1]
          for i in range(2,len(last_layer_input_size)):
            length *= last_layer_input_size[i]
          params = [1,length] + [0,0,1,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          
          #加入 #3 avgpool
          
          pooling_params = [last_layer_input_size[2],last_layer_input_size[2], 1,1, 0,0, 1,1, 1]
          params = last_layer_input_size + [0,0,0,0] + pooling_params
          output_size = self.get_output_size(params,7,last_layer_input_size)
          params = last_block_input_size + output_size + pooling_params
          self.layer_parameters += params #加入参数向量中
          link_list = [self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [7] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          
        else:
          #生成全连接层
          # 几率加入avgpool ,reshape为 [1,channel,1,1]
          if random.randint(0,10)<3:
            pooling_params = [last_block_input_size[2],last_block_input_size[2], 1,1, 0,0, 1,1, 1]
            params = last_block_input_size + [0,0,0,0] + pooling_params
            output_size = self.get_output_size(params,7,last_block_input_size)
            params = last_block_input_size + output_size + pooling_params
            self.layer_parameters += params #加入参数向量中
            link_list = [self.layer_num-1]
            self.layer_link += self.get_link_vector(link_list,self.layer_num)
            self.layer_id += [7] # 加入层id列表中
            self.layer_num += 1
            last_layer_input_size = output_size
            last_block_input_size = last_layer_input_size
          
          #加入全连接层
          input_length = last_block_input_size[1] * last_block_input_size[2] * last_block_input_size[3]
          params = self.make_layer(21,[1,input_length],[1,1000])
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
      in_channel,in_height = input_size[1],input_size[2]
      out_height = output_size[2]
      #print(input_size,output_size)
      if in_height == 224:
        choose = 0
        if out_height == 112:
          choose = 11
          return_index,out_size = self.make_conv_bn(last_block_index,input_size,output_size)
          
        elif out_height == 56:
          
          if random.randint(0,100) < 50:
            choose = 21
            return_index,out_size = self.make_inceptionV1_pre(last_block_index,input_size,output_size)
          else:
            choose = 22
            return_index,out_size = self.make_resnet_pre(last_block_index,input_size,output_size)
        # print('==out_height:',choose)
        
      elif out_height == in_height:
        
        choose = random.randint(0,96)
        # if choose > 48 and choose < 64 and in_channel > 192:
        #   choose = choose - 16
        # print(1,choose)
        if choose < 16:
          choose = 1
          return_index,out_size = self.make_conv_dw(last_block_index,input_size,output_size)
        elif choose < 32:
          choose = 2
          return_index,out_size = self.make_InversedResidual(last_block_index,input_size,output_size)
        elif choose < 48:
          choose = 3
          return_index,out_size = self.make_resnet18_block(last_block_index,input_size,output_size)
        elif choose < 64:
          choose = 4
          return_index,out_size = self.make_MobileNetV3_block(last_block_index,input_size,output_size)
        elif choose < 80:
          choose = 5
          return_index,out_size = self.make_resnet50_block(last_block_index,input_size,output_size)
        else:
          choose = 6
          return_index,out_size = self.make_inceptionV1_block(last_block_index,input_size,output_size)
        # print('==choose:',choose)
      
      elif in_height == 2*out_height:
        choose = random.randint(0,112)
        # if choose > 80 and in_channel < 192:
        #   choose = choose - 16
        # if choose > 64 and choose < 80 and in_channel > 192:
        #   choose = choose - 16
        # print(2,choose)
        if choose < 16:
          choose = 1
          return_index,out_size = self.make_conv_dw(last_block_index,input_size,output_size)
        elif choose < 32:
          choose = 2
          return_index,out_size = self.make_InversedResidual(last_block_index,input_size,output_size)
        elif choose < 48:
          choose = 3
          return_index,out_size = self.make_resnet18_block(last_block_index,input_size,output_size)
        elif choose < 64:
          choose = 4
          return_index,out_size = self.make_MobileNetV3_block(last_block_index,input_size,output_size)
        elif choose < 80:
          choose = 5
          return_index,out_size = self.make_inceptionV1_block2(last_block_index,input_size,output_size)
        elif choose < 96:
          choose = 6
          return_index,out_size = self.make_resnet50_block(last_block_index,input_size,output_size)
        else :
          choose = 7
          return_index,out_size = self.make_inceptionV1_block(last_block_index,input_size,output_size)
        # print('2*choose:',choose)
        
      return self.layer_num - 1,out_size #返回块输出元素
   
    def make_conv_bn(self,last_block_index,input_size,final_output_size):
      '''
      实现MobileNet-v1 中第一层的标准卷积
      #nn.Conv2d(inp,oup,kernel_size=3,stride = 2,padding=1,bias = False),
      #nn.BatchNorm2d(oup),
      #nn.ReLU(inplace = True))
      Input_h = 2* Output_h
      '''
      last_layer_input_size = input_size
      #加入 #1 conv2d
      in_channel = input_size[1]
      out_channel = random.randint(32,48)
      conv_params = [in_channel,out_channel, 3,3, 2,2, 1,1, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      #print('make_conv_bn',last_layer_input_size)
      return self.layer_num - 1, last_layer_input_size #返回该线性序列的最后一个节点的索引号

    def make_conv_dw(self,last_block_index,input_size,final_output_size):
      '''
      实现MobileNet-v1 中的深度卷积
      stride = 1  → Input_h = Output_h
      stride = 2  → Input_h = 2*Output_h
      #nn.Conv2d(inp,inp,kernel_size=3,stride=1&2,padding=1,groups = inp,bias = False),
      #nn.BatchNorm2d(inp),
      #nn.ReLU(inplace = True),
                    
      #nn.Conv2d(inp,oup,kernel_size=1,stride=1,padding=0,bias = False),
      #nn.BatchNorm2d(oup),
      #nn.ReLU(inplace = True)

      Input_h = 2 * Output_h
      '''
      last_layer_input_size = input_size
      in_channel,in_height = input_size[1],input_size[2]
      out_channel = random.randint(in_channel,in_channel*2)
      if out_channel > 1000:
        out_channel = self.prob_random([int(in_channel/2),int(in_channel/3)],[0.8,0.2])
      #加入 #1 conv2d
      out_height = final_output_size[2]
      if in_height == out_height:
        stride = 1
      elif  in_height == 2 *out_height:
        stride = 2
      else:
        print('wrong')
      conv_params = [in_channel,in_channel, 3,3, stride,stride, 1,1, 1,1, in_channel]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      

      #加入 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1

      #加入 #4 conv2d
      conv_params = [in_channel,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
      params = last_layer_input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 #5 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #6 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1

      #print('make_conv_dw',last_layer_input_size)
      return self.layer_num - 1,last_layer_input_size #返回该线性序列的最后一个节点的索引号

    def make_InversedResidual(self,last_block_index,input_size,final_output_size):
      '''
      已改
      out_channel < 400可用
      实现MobileNet-v2 中的InversedResidual
      
      stride = 2  → Input_h = 2*Output_h
      #nn.Conv2d(inp, inp, kernel_size=3, stride=2, padding=1, groups=inp, bias=False),
      #nn.BatchNorm2d(inp),
      #nn.ReLU6(inplace=True),
      #nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
      #nn.BatchNorm2d(oup),
      
      stride = 1  → Input_h = Output_h
      
      #nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
      #nn.BatchNorm2d(hidden_dim),
      #nn.ReLU6(inplace=True),
      #nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
      #nn.BatchNorm2d(hidden_dim),
      #nn.ReLU6(inplace=True),
      #nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
      #nn.BatchNorm2d(oup),
      
      '''
      
      last_layer_input_size = input_size
      in_channel,in_height = input_size[1],input_size[2]
      out_channel = random.randint(in_channel,in_channel*2)
      if out_channel > 1000:
        out_channel = self.prob_random([int(in_channel/2),int(in_channel/3)],[0.8,0.2])
      out_height = final_output_size[2]
      if in_height == out_height:
        stride = 1
        #加入 #1 conv2d
        res = self.layer_num - 1
        hidden_dim = in_channel * 6
        if hidden_dim > 1000:
          hidden_dim = in_channel * 3
        conv_params = [in_channel,hidden_dim, 1,1, 1,1, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [hidden_dim,hidden_dim, 3,3, 1,1, 1,1, 1,1, hidden_dim]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
                
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #6 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
                
        #加入 #7 conv2d
        conv_params = [hidden_dim,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #8 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入残差连接
        if in_channel == out_channel :
          add_index_list = [res,self.layer_num-1]
          params = self.make_layer(24,last_layer_input_size,add_num=len(add_index_list))
          self.layer_parameters += params #加入参数向量中
          self.layer_link += self.get_link_vector(add_index_list,self.layer_num)
          self.layer_id += [24] # 加入层id列表中
          self.layer_num += 1
     
      elif  in_height == 2 *out_height:
        stride = 2
        #加入 #1 conv2d
        conv_params = [in_channel,in_channel, 3,3, stride,stride, 1,1, 1,1, in_channel]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [in_channel,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
      else:
        print('wrong')
      
      #print('make_InversedResidual',last_layer_input_size)
      return self.layer_num - 1,last_layer_input_size #返回该线性序列的最后一个节点的索引号

    def make_MobileNetV3_block(self,last_block_index,input_size,final_output_size):
      '''
      IN = 2 OUT
      (conv): Sequential(
        (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
        (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SEModule(
          (avg_pool): AdaptiveAvgPool2d(output_size=1)
          (fc): Sequential(
            (0): Linear(in_features=16, out_features=4, bias=False)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=4, out_features=16, bias=False)
            (3): Hsigmoid()
          )
        )
        (6): ReLU(inplace=True)
        (7): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      IN = OUT
      (conv): Sequential(
        (0): Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
        (4): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): Identity()
        (6): ReLU(inplace=True)
        (7): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (8): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      '''
      
      
      last_layer_input_size = input_size
      in_channel,in_height = input_size[1],input_size[2]
      if random.randint(0,100) < 20:
        out_channel = int(in_channel*2)
      else:
        out_channel = random.randint(in_channel,int(in_channel*1.5))
      if out_channel > 1000:
        out_channel = self.prob_random([int(in_channel/2),int(in_channel/3)],[0.8,0.2])
      out_height = final_output_size[2]
      hidden_dim = random.randint(in_channel*3,in_channel*6)
      if hidden_dim > 1000:
        hidden_dim = in_channel 
      if in_height == out_height:
        stride = 1
        #加入 #1 conv2d
        res = self.layer_num - 1
        conv_params = [in_channel,hidden_dim, 1,1, 1,1, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [hidden_dim,hidden_dim, 3,3, 1,1, 1,1, 1,1, hidden_dim]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
                
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        before_le_input_size = last_layer_input_size
        before_le = self.layer_num - 1
        #加入 SEModule
        if random.randint(0,100)<50:
          #加入 Avgpool
          kernel_size = last_layer_input_size[2]
          pooling_params = [kernel_size,kernel_size, 1,1, 0,0, 1,1, 1]
          params = last_layer_input_size + [0,0,0,0] + pooling_params
          output_size = self.get_output_size(params,7,last_layer_input_size)
          params = last_layer_input_size + output_size + pooling_params
          self.layer_parameters += params #加入参数向量中
          link_list = [self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [7] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          
          #加入 Linear
          linear_hidden = int(hidden_dim/4)
          input_length = last_layer_input_size[1] * last_layer_input_size[2] * last_layer_input_size[3]
          params = self.make_layer(21,[1,input_length],[1,linear_hidden])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          #加入ReLU
          length = linear_hidden
          params = [1,length] + [0,0,1,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          
          #加入 Linear
          params = self.make_layer(21,[1,linear_hidden],[1,hidden_dim])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          #加入Sigmoid
          length = hidden_dim
          params = [1,hidden_dim] + [1,0,0,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
        
        #加入 #6 ReLU
        length = before_le_input_size[1]
        for i in range(2,len(before_le_input_size)):
          length *= before_le_input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[before_le]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
                
        #加入 #7 conv2d
        conv_params = [hidden_dim,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = before_le_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,before_le_input_size)
        params = before_le_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #8 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入残差连接
        if in_channel == out_channel :
          add_index_list = [res,self.layer_num-1]
          params = self.make_layer(24,last_layer_input_size,add_num=len(add_index_list))
          self.layer_parameters += params #加入参数向量中
          self.layer_link += self.get_link_vector(add_index_list,self.layer_num)
          self.layer_id += [24] # 加入层id列表中
          self.layer_num += 1
     
      elif  in_height == 2 *out_height:
        stride = 2
        #加入 #1 conv2d
        conv_params = [in_channel,hidden_dim, 1,1, 1,1, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [hidden_dim,hidden_dim, 3,3, 2,2, 1,1, 1,1, hidden_dim]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        before_le_input_size = last_layer_input_size
        before_le = self.layer_num - 1
        #加入 SEModule
        if random.randint(0,100)<50:
          #加入 Avgpool
          kernel_size = last_layer_input_size[2]
          pooling_params = [kernel_size,kernel_size, 1,1, 0,0, 1,1, 1]
          params = last_layer_input_size + [0,0,0,0] + pooling_params
          output_size = self.get_output_size(params,7,last_layer_input_size)
          params = last_layer_input_size + output_size + pooling_params
          self.layer_parameters += params #加入参数向量中
          link_list = [self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [7] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          #加入 Linear
          linear_hidden = int(hidden_dim/4)
          input_length = last_layer_input_size[1] * last_layer_input_size[2] * last_layer_input_size[3]
          params = self.make_layer(21,[1,input_length],[1,linear_hidden])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          #加入ReLU
          length = linear_hidden
          params = [1,length] + [0,0,1,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          
          #加入 Linear
          params = self.make_layer(21,[1,linear_hidden],[1,hidden_dim])
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [21] # 加入层id列表中
          self.layer_num += 1
          
          #加入Sigmoid
          length = hidden_dim
          params = [1,hidden_dim] + [1,0,0,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
        
        #加入 #6 ReLU
        length = before_le_input_size[1]
        for i in range(2,len(before_le_input_size)):
          length *= before_le_input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[before_le]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
                
        #加入 #7 conv2d
        conv_params = [hidden_dim,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = before_le_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,before_le_input_size)
        params = before_le_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #8 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
      else:
        print('wrong')
      
      #print('make_InversedResidual',last_layer_input_size)
      return self.layer_num - 1,last_layer_input_size #返回该线性序列的最后一个节点的索引号

    def make_resnet_pre(self,last_block_index,input_size,final_output_size):
      '''
      实现ResNet最初的pre层
      (0): Conv2d(3, hidden, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))       
      (1): BatchNorm2d(hidden)    
      (2): ReLU(inplace=True)
      (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
      out:torch.Size([1, 64, 56, 56])
      '''
      last_layer_input_size = input_size
      #加入 #1 conv2d
      in_channel = input_size[1]
      out_channel = 64
      conv_params = [in_channel,out_channel, 7,7, 2,2, 3,3, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      #加入 #4 pooling
      pooling_params = [3,3, 2,2, 1,1, 1,1, 0]
      params = last_layer_input_size + [0,0,0,0] + pooling_params
      output_size = self.get_output_size(params,7,last_layer_input_size)
      params = last_layer_input_size + output_size + pooling_params
      self.layer_parameters += params #加入参数向量中
      link_list = [self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [7] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #('resnetpre',last_layer_input_size)
      return self.layer_num - 1,last_layer_input_size #返回该线性序列的最后一个节点的索引号
    
    def make_resnet18_block(self,last_block_index,input_size,final_output_size):
      '''
      Resnet 残差模块，包括左右两部分
      stride=2 → in_height == 2*out_height 包括左右两部分
      (left): Sequential(
        (0): Conv2d(inp, otp, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(otp, otp, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (right): Sequential(
        (0): Conv2d(inp, otp, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        
      stride=1 → in_height == out_height 包括左
      (left): Sequential(
        (0): Conv2d(inp, inp, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(inp, inp, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      #可无
      (right): Sequential(
        (0): Conv2d(inp, inp, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      
      (relu): ReLU(inplace=True)
      '''
      in_channel,in_height = input_size[1],input_size[2]
      out_channel = self.prob_random([in_channel,int(in_channel*2)],[0.8,0.2])
      if out_channel > 1000:
        out_channel = self.prob_random([in_channel,int(in_channel/2),int(in_channel/3)],[0.05,0.7,0.25])
      out_height = final_output_size[2]
      if in_height == out_height:
        
        right_res_id = self.layer_num - 1
        #加入 #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,in_channel, 3,3, 1,1, 1,1, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [in_channel,in_channel, 3,3, 1,1, 1,1, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 
        
        #print('left',last_layer_input_size)
      
        
        left_res_id = self.layer_num - 1
        
        if(random.randint(1,100) < 30):
          #加入 right #1 conv2d
          in_channel = input_size[1]
          conv_params = [in_channel,in_channel, 1,1, 1,1, 0,0, 1,1, 1]
          params = input_size + [0,0,0,0] + conv_params
          output_size = self.get_output_size(params,1,input_size)
          params = input_size + output_size + conv_params
          self.layer_parameters += params #加入参数向量中
          link_list=[last_block_index]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [1] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          
          #加入 right #2 BN
          params = self.make_layer(16,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [16] # 加入层id列表中
          self.layer_num += 1
          
          #加入 right #3 ReLU
          length = last_layer_input_size[1]
          for i in range(2,len(input_size)):
            length *= input_size[i]
          params = [1,length] + [0,0,1,0]
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [22] # 加入层id列表中
          self.layer_num += 1
          
          right_res_id = self.layer_num - 1
          #print('right',last_layer_input_size)
        
      elif in_height == 2 * out_height:
        #加入 left #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,out_channel, 3,3, 2,2, 1,1, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 left #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 left #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 left #4 conv2d
        conv_params = [out_channel,out_channel, 3,3, 1,1, 1,1, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 left #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 
        
        left_res_id = self.layer_num - 1
        #print('left',last_layer_input_size)
        
        #加入 right #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,out_channel, 1,1, 2,2, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 right #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 right #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        #print('right',last_layer_input_size)
        
        right_res_id = self.layer_num - 1
        
      else:
        print('wrong')
      
      #加入残差连接
      add_index_list = [left_res_id,right_res_id]
      params = self.make_layer(24,last_layer_input_size,add_num=len(add_index_list))
      self.layer_parameters += params #加入参数向量中
      self.layer_link += self.get_link_vector(add_index_list,self.layer_num)
      self.layer_id += [24] # 加入层id列表中
      self.layer_num += 1
      
      #加入ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1

      #print('resnet18block',last_layer_input_size)
      return self.layer_num - 1, last_layer_input_size
  
    def make_resnet50_block(self,last_block_index,input_size,final_output_size):
      '''
      已改
      Resnet 残差模块，包括左右两部分
      stride=2 → in_height == 2*out_height 包括左右两部分
      (bottleneck): Sequential(
        (0): Conv2d(inp, hidden, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(hidden, otp, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (downsample): Sequential(
        (0): Conv2d(inp, otp, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(otp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
      
      stride=1 → in_height == out_height 包括左
      (left): Sequential(
        (0): Conv2d(inp, hidden, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(hidden, inp, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      #可无 10%几率有
      (downsample): Sequential(
        (0): Conv2d(inp, inp, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
      '''
      in_channel,in_height = input_size[1],input_size[2]
      
      out_channel = self.prob_random([in_channel,int(in_channel*2)],[0.8,0.2])
      if out_channel > 1000:
        out_channel = self.prob_random([in_channel,int(in_channel/2),int(in_channel/3)],[0.05,0.7,0.25])
        
      out_height = final_output_size[2]
      if in_height == out_height:
        right_res_id = self.layer_num - 1
        hidden = self.prob_random([int(in_channel*0.5),int(in_channel * 0.25)],[0.5,0.5])
        #加入 #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,hidden, 1,1, 1,1, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [hidden,hidden, 3,3, 1,1, 1,1, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 

        #加入 #6 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #7 conv2d
        conv_params = [hidden,in_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 #8 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 
        
        left_res_id = self.layer_num - 1
        
        if(random.randint(1,100) < 10):
          
          #加入 right #1 conv2d
          in_channel = input_size[1]
          conv_params = [in_channel,in_channel, 1,1, 1,1, 0,0, 1,1, 1]
          params = input_size + [0,0,0,0] + conv_params
          output_size = self.get_output_size(params,1,input_size)
          params = input_size + output_size + conv_params
          self.layer_parameters += params #加入参数向量中
          link_list=[last_block_index]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [1] # 加入层id列表中
          self.layer_num += 1
          last_layer_input_size = output_size
          
          #加入 right #2 BN
          params = self.make_layer(16,last_layer_input_size)
          self.layer_parameters += params #加入参数向量中
          link_list=[self.layer_num-1]
          self.layer_link += self.get_link_vector(link_list,self.layer_num)
          self.layer_id += [16] # 加入层id列表中
          self.layer_num += 1
          
          right_res_id = self.layer_num - 1
        
      elif in_height == 2 * out_height:
        right_res_id = self.layer_num - 1
        
        hidden = self.prob_random([int(in_channel*0.5),int(in_channel * 0.33),int(in_channel * 0.25)],[0.8,0.1,0.1])
        #加入 #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,hidden, 1,1, 1,1, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        
        #加入 #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #3 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #4 conv2d
        conv_params = [hidden,hidden, 3,3, 2,2, 1,1, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 #5 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 

        #加入 #6 ReLU
        length = last_layer_input_size[1]
        for i in range(2,len(input_size)):
          length *= input_size[i]
        params = [1,length] + [0,0,1,0]
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [22] # 加入层id列表中
        self.layer_num += 1
        
        #加入 #7 conv2d
        conv_params = [hidden,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
        params = last_layer_input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,last_layer_input_size)
        params = last_layer_input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size       
        
        #加入 #8 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1 
        
        left_res_id = self.layer_num - 1
        
          
        #加入 right #1 conv2d
        in_channel = input_size[1]
        conv_params = [in_channel,out_channel, 1,1, 2,2, 0,0, 1,1, 1]
        params = input_size + [0,0,0,0] + conv_params
        output_size = self.get_output_size(params,1,input_size)
        params = input_size + output_size + conv_params
        self.layer_parameters += params #加入参数向量中
        link_list=[last_block_index]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [1] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
          
        #加入 right #2 BN
        params = self.make_layer(16,last_layer_input_size)
        self.layer_parameters += params #加入参数向量中
        link_list=[self.layer_num-1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [16] # 加入层id列表中
        self.layer_num += 1
          
        right_res_id = self.layer_num - 1
        
      else:
        print('wrong')
      
      #加入残差连接
      add_index_list = [left_res_id,right_res_id]
      params = self.make_layer(24,last_layer_input_size,add_num=len(add_index_list))
      self.layer_parameters += params #加入参数向量中
      self.layer_link += self.get_link_vector(add_index_list,self.layer_num)
      self.layer_id += [24] # 加入层id列表中
      self.layer_num += 1

      #print('resnet50block',last_layer_input_size)
      return self.layer_num - 1,last_layer_input_size

    def make_inceptionV1_pre(self,last_block_index,input_size,final_output_size):
      '''
      inceptionV1 初始层
      nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
      nn.BatchNorm2d(64),
      [1, 64, 56, 56]
      '''
      
      last_layer_input_size = input_size
      #加入 #1 conv2d
      in_channel = input_size[1]
      out_channel = random.randint(56,72)
      conv_params = [in_channel,out_channel, 7,7, 2,2, 3,3, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #3 pooling
      pooling_params = [3,3, 2,2, 1,1, 1,1, 0]
      params = last_layer_input_size + [0,0,0,0] + pooling_params
      output_size = self.get_output_size(params,7,last_layer_input_size)
      params = last_layer_input_size + output_size + pooling_params
      self.layer_parameters += params #加入参数向量中
      link_list = [self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [7] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      
      #加入 #4 conv2d
      in_channel = input_size[1]
      conv_params = [out_channel,out_channel, 1,1, 1,1, 0,0, 1,1, 1]
      params = last_layer_input_size + final_output_size + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      
      #加入 #5 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #print('inceptionpre',last_layer_input_size)
      return self.layer_num - 1, last_layer_input_size #返回该线性序列的最后一个节点的索引号

    def make_inceptionV1_block2(self,last_block_index,input_size,final_output_size):
      '''
      inceptionV1 block2
      
      nn.Conv2d(in_channels=inp, out_channels=otp, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(otp),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      
      '''
      #加入 #1 conv2d
      in_channel = input_size[1]
      out_channel = self.prob_random([int(in_channel),int(in_channel*2),int(in_channel*3)],[0.8,0.1,0.1])
      if out_channel > 1000:
        out_channel = self.prob_random([int(in_channel/2),int(in_channel/3)],[0.8,0.2])
      
      conv_params = [in_channel,out_channel, 3,3, 1,1, 1,1, 1,1, 1]
      params = input_size + final_output_size + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 #3 pooling
      pooling_params = [3,3, 2,2, 1,1, 1,1, 0]
      params = last_layer_input_size + [0,0,0,0] + pooling_params
      output_size = self.get_output_size(params,7,last_layer_input_size)
      params = last_layer_input_size + output_size + pooling_params
      self.layer_parameters += params #加入参数向量中
      link_list = [self.layer_num-1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [7] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      
      #print('make_inceptionV1_block2',last_layer_input_size)
      return self.layer_num - 1, last_layer_input_size

    def make_inceptionV1_block(self,last_block_index,input_size,final_output_size):
      '''
      >200可用
      InceptionV1Module
      self.branch1 = ConvBNReLU(in_channels=inp,out_channels=out_channels1,kernel_size=1)

      self.branch2 = nn.Sequential(ConvBNReLU(in_channels=inp,out_channels=out_channels2reduce,kernel_size=1),
                                    ConvBNReLU(in_channels=out_channels2reduce,out_channels=out_channels2,kernel_size=3))

      self.branch3 = nn.Sequential(ConvBNReLU(in_channels=inp, out_channels=out_channels3reduce, kernel_size=1),
                                     ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5))
        
      self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                                    ConvBNReLU(in_channels=inp, out_channels=out_channels4, kernel_size=1))

      ConvBNReLU(in_channels,out_channels,kernel_size):
      return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
      )
      
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 50%概率加
      '''
      
      in_channel = input_size[1]
      out_channel = final_output_size[1]
      
      # block_type = self.prob_random([1,2,3,4,5,6,7,8,9],[0.08,0.06,0.11,0.14,0.16,0.12,0.08,0.15,0.1])
      if in_channel > 800:
        in_channel = int(in_channel/2)
      out_channel_1 = random.randint(int(in_channel/4),int(in_channel/2))
        
      out_channel_2 = random.randint(int(in_channel*0.33),int(in_channel*0.66))
      out_channel_2_hidden = self.prob_random([int(out_channel_2*0.2),int(out_channel_2*0.5)],[0.5,0.5])
        
      out_channel_3 = random.randint(int(in_channel*0.1),int(in_channel*0.375))
      out_channel_3_hidden = self.prob_random([int(out_channel_3/2),int(out_channel_3/3),int(out_channel_3/4)],[0.33,0.33,0.34])
        
      out_channel_4 = random.randint(int(in_channel/4),int(in_channel/2))
      #1.333
      # if block_type == 1:
                
      #   out_channel_1 = int(in_channel/3)
        
      #   out_channel_2 = int(in_channel/3) * 2
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel/6)
      #   out_channel_3_hidden = int(out_channel_3/2)
        
      #   out_channel_4 = int(in_channel/6)
      # #1.875
      # elif block_type == 2:
                
      #   out_channel_1 = int(in_channel/2)
        
      #   out_channel_2 = int(in_channel * 0.75)
      #   out_channel_2_hidden = int(in_channel/2)
        
      #   out_channel_3 = int(in_channel* 0.375)
      #   out_channel_3_hidden = int(out_channel_3/3)
        
      #   out_channel_4 = int(in_channel/4)
      # #1.067
      # elif block_type == 3:
                
      #   out_channel_1 = int(in_channel*0.4)
        
      #   out_channel_2 = int(in_channel * 0.43)
      #   out_channel_2_hidden = int(in_channel * 0.2)
        
      #   out_channel_3 = int(in_channel* 0.1)
      #   out_channel_3_hidden = int(out_channel_3/3)
        
      #   out_channel_4 = int(in_channel * 0.133)
      # #1
      # elif block_type == 4:
                
      #   out_channel_1 = int(in_channel*0.3125)
        
      #   out_channel_2 = int(in_channel * 0.4375)
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel* 0.125)
      #   out_channel_3_hidden = int(out_channel_3 * 0.375)
        
      #   out_channel_4 = int(in_channel*0.125)
      # #1
      # elif block_type == 5:
                
      #   out_channel_1 = int(in_channel*0.25)
        
      #   out_channel_2 = int(in_channel * 0.5)
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel* 0.125)
      #   out_channel_3_hidden = int(out_channel_3 * 0.375)
        
      #   out_channel_4 = int(in_channel*0.125)
      # #1.03125
      # elif block_type == 6:
                
      #   out_channel_1 = int(in_channel*0.21875)
        
      #   out_channel_2 = int(in_channel * 0.5625)
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel* 0.125)
      #   out_channel_3_hidden = int(out_channel_3 /2)
        
      #   out_channel_4 = int(in_channel*0.125)
      # #1.5758
      # elif block_type == 7:
                
      #   out_channel_1 = int(in_channel*0.485)
        
      #   out_channel_2 = int(in_channel * 0.606)
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel* 0.2424)
      #   out_channel_3_hidden = int(out_channel_3 /4)
        
      #   out_channel_4 = int(in_channel*0.2424)
      # #1
      # elif block_type == 8:
                
      #   out_channel_1 = int(in_channel*0.3077)
        
      #   out_channel_2 = int(in_channel * 0.3846)
      #   out_channel_2_hidden = int(out_channel_2/2)
        
      #   out_channel_3 = int(in_channel* 0.1538)
      #   out_channel_3_hidden = int(out_channel_3 /4)
        
      #   out_channel_4 = int(in_channel*0.1538)
      # #1.2367
      # elif block_type == 9:
                
        # out_channel_1 = int(in_channel*0.4615)
        
        # out_channel_2 = int(in_channel * 0.4615)
        # out_channel_2_hidden = int(out_channel_2/2)
        
        # out_channel_3 = int(in_channel* 0.1538)
        # out_channel_3_hidden = int(out_channel_3 /3)
        
        # out_channel_4 = int(in_channel*0.1538)
      branch1_index = last_block_index
      #加入 Branch1 #1 conv2d
      in_channel = input_size[1]
      conv_params = [in_channel,out_channel_1, 1,1, 1,1, 0,0, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch1 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch1 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch1_res_index = self.layer_num - 1
      
      #加入 Branch2 #1 conv2d
      in_channel = input_size[1]
      out_channel = final_output_size[1]
      conv_params = [in_channel,out_channel_2_hidden, 1,1, 1,1, 0,0, 1,1, 1]
      params = last_layer_input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch2 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch2 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      #加入 Branch2 #4 conv2d
      conv_params = [out_channel_2_hidden,out_channel_2, 3,3, 1,1, 1,1, 1,1, 1]
      params = last_layer_input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch2 #5 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch2 #6 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch2_res_index = self.layer_num - 1
      
      #加入 Branch3 #1 conv2d
      in_channel = input_size[1]
      out_channel = final_output_size[1]
      conv_params = [in_channel,out_channel_3_hidden, 1,1, 1,1, 0,0, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch3 #2 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch3 #3 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      #加入 Branch3 #4 conv2d
      conv_params = [out_channel_3_hidden,out_channel_3, 5,5, 1,1, 2,2, 1,1, 1]
      params = last_layer_input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch3 #5 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch3 #6 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch3_res_index = self.layer_num - 1
      
      #加入 Branch4 #1 pooling
      pooling_params = [3,3, 1,1, 1,1, 1,1, 0]
      params = input_size + [0,0,0,0] + pooling_params
      output_size = self.get_output_size(params,7,input_size)
      params = input_size + output_size + pooling_params
      self.layer_parameters += params #加入参数向量中
      link_list = [last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [7] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      
      #加入 Branch4 #2 conv2d
      in_channel = input_size[1]
      conv_params = [in_channel,out_channel_4, 3,3, 1,1, 1,1, 1,1, 1]
      params = last_layer_input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,last_layer_input_size)
      params = last_layer_input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 Branch4 #3 BN
      params = self.make_layer(16,last_layer_input_size)
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [16] # 加入层id列表中
      self.layer_num += 1

      #加入 Branch4 #4 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch4_res_index = self.layer_num - 1
      
      
      final_out_channels = out_channel_1 + out_channel_2 + out_channel_3 + out_channel_4
      output_size = [1,final_out_channels,last_layer_input_size[2],last_layer_input_size[3]]
      #加入concat
      layer_output_index_list = [branch1_res_index,branch2_res_index,branch3_res_index,branch4_res_index]
      params = self.make_layer(23,output_size,out_channels=final_out_channels)
      self.layer_parameters += params #加入参数向量中
      self.layer_link += self.get_link_vector(layer_output_index_list,self.layer_num)
      self.layer_id += [23] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size
      
      #判断是否加入Pooling
      in_height = input_size[2]
      out_height = final_output_size[2]
      if in_height == out_height:
        #print('==make_inceptionV1_block',last_layer_input_size)
        return self.layer_num - 1,last_layer_input_size
      elif in_height == 2*out_height:
        #加入Pooling
        pooling_params = [3,3, 2,2, 1,1, 1,1, 0]
        params = last_layer_input_size + [0,0,0,0] + pooling_params
        output_size = self.get_output_size(params,7,last_layer_input_size)
        params = last_layer_input_size + output_size + pooling_params
        self.layer_parameters += params #加入参数向量中
        link_list = [self.layer_num - 1]
        self.layer_link += self.get_link_vector(link_list,self.layer_num)
        self.layer_id += [7] # 加入层id列表中
        self.layer_num += 1
        last_layer_input_size = output_size
        #print('2*make_inceptionV1_block',last_layer_input_size)
        return self.layer_num - 1,last_layer_input_size
      else:
        print('Inception Wrong')
  
    def make_squeeze_fire_block(self,last_block_index,input_size,final_output_size):
      '''
      加入suqueeze fire层
      (8): Fire(
        (squeeze): Sequential(
          (0): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
          (1): ReLU(inplace=True)
        )
        (expand_1): Sequential(
          (0): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
          (1): ReLU(inplace=True)
        )
        (expand_3): Sequential(
          (0): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
        )
      )
      '''
      in_channel = input_size[1]
      squeeze_hidden_channel = self.prob_random([int(in_channel/4), int(in_channel/6), int(in_channel/8)],[0.05,0.2,0.75])
      if squeeze_hidden_channel > 300:
        squeeze_hidden_channel = self.prob_random([squeeze_hidden_channel,int(squeeze_hidden_channel/2),int(squeeze_hidden_channel/3)],[0.1,0.7,0.2])
      expand_1_out_channel = squeeze_hidden_channel * 4
      expand_2_out_channel = expand_1_out_channel
      
      #加入 squeeze #1 conv2d
      in_channel = input_size[1]
      conv_params = [in_channel,squeeze_hidden_channel, 1,1, 1,1, 0,0, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[last_block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 squeeze #2 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      block_index = self.layer_num - 1
      
      #加入 expand_1 #1 conv2d
      in_channel = input_size[1]
      conv_params = [squeeze_hidden_channel,expand_1_out_channel, 1,1, 1,1, 0,0, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 expand_1 #2 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch1_res_index=self.layer_num - 1
      
      #加入 expand_3 #1 conv2d
      conv_params = [squeeze_hidden_channel,expand_2_out_channel, 3,3, 1,1, 1,1, 1,1, 1]
      params = input_size + [0,0,0,0] + conv_params
      output_size = self.get_output_size(params,1,input_size)
      params = input_size + output_size + conv_params
      self.layer_parameters += params #加入参数向量中
      link_list=[block_index]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [1] # 加入层id列表中
      self.layer_num += 1
      last_layer_input_size = output_size

      #加入 expand_3 #2 ReLU
      length = last_layer_input_size[1]
      for i in range(2,len(input_size)):
        length *= input_size[i]
      params = [1,length] + [0,0,1,0]
      self.layer_parameters += params #加入参数向量中
      link_list=[self.layer_num - 1]
      self.layer_link += self.get_link_vector(link_list,self.layer_num)
      self.layer_id += [22] # 加入层id列表中
      self.layer_num += 1
      
      branch2_res_index=self.layer_num - 1
      
      final_out_channels = expand_1_out_channel + expand_2_out_channel
      output_size = [1,final_out_channels,last_layer_input_size[2],last_layer_input_size[3]]
      #加入concat
      layer_output_index_list = [branch1_res_index,branch2_res_index]
      params = self.make_layer(23,output_size,out_channels=final_out_channels)
      self.layer_parameters += params #加入参数向量中
      self.layer_link += self.get_link_vector(layer_output_index_list,self.layer_num)
      self.layer_id += [23] # 加入层id列表中
      self.layer_num += 1
      
      #print('fire',output_size)
      return self.layer_num - 1,output_size
  
    def get_output_size(self,parameters,layer_id,input_size):
        
        if(layer_id == 0):
            in_channels,out_channels,kernel_size,stride,padding,dilation,groups = parameters[-7:]


        elif(layer_id == 1):
            batch_size,channel,input_height,input_width = input_size
            in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,groups = parameters[-11:]
            out_height = math.floor((input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1)
            out_width = math.floor((input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1)
            return [batch_size,out_channels,out_height,out_width]
            
        elif(layer_id == 2):
            in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_width,dilation_height,groups = parameters[-15:]
          

        elif(layer_id == 3):
            in_channels,out_channels,kernel_size,stride,padding,output_padding,dilation,groups = parameters[-8:]

        elif(layer_id == 4):
            in_channels,out_channels,kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,output_padding_height,output_padding_width,dilation,groups = parameters[-12:]

            
        elif(layer_id == 5):
            in_channels,out_channels,kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,\
          padding_height,padding_width,output_padding_depth,output_padding_height,output_padding_width,dilation,groups = parameters[-16:]

        elif(layer_id == 6):
            kernel_size,stride,padding,dilation,pool_type = parameters[-5:]
                   
        elif(layer_id == 7):
            
            batch_size,channel,input_height,input_width = input_size
            kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width,dilation_height,dilation_width,pool_type = parameters[-9:]
            out_height = math.floor((input_height + 2*padding_height - dilation_height*(kernel_size_height-1) - 1)/(stride_height) + 1)
            out_width = math.floor((input_width + 2*padding_width - dilation_width*(kernel_size_width-1) - 1)/(stride_width) + 1)
            return [batch_size,channel,out_height,out_width]

            
        elif(layer_id == 8):
            kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width,dilation_depth,dilation_height,dilation_width,pool_type = parameters[-13:]

            
        elif(layer_id == 9):
            kernel_size,stride,padding = parameters[-3:]

            
        elif(layer_id == 10):
            kernel_size_height,kernel_size_width,stride_height,stride_width,padding_height,padding_width = parameters[-6:]

        elif(layer_id == 11):
            kernel_size_depth,kernel_size_height,kernel_size_width,stride_depth,stride_height,stride_width,padding_depth,padding_height,padding_width = parameters[-9:]

        elif(layer_id == 12):
            output_size_L,pool_type = parameters[-2:]

        elif(layer_id == 13):
            output_size_H,output_size_W,pool_type = parameters[-3:]

        elif(layer_id == 14):
            output_size_D,output_size_H,output_size_W,pool_type = parameters[-4:]
        
        elif(layer_id == 15):
            num_features = parameters[-1:][0]

        elif(layer_id == 16):
            num_features = parameters[-1:][0]

        elif(layer_id == 17):
            num_features = parameters[-1:][0]
        
        elif(layer_id == 18):
            probability = parameters[-1:][0]

        elif(layer_id == 19):
            probability = parameters[-1:][0]
          
        elif(layer_id == 20):
            probability = parameters[-1:][0]
          
        elif(layer_id == 21):
            batch_size = input_size[0]
            input_length,output_length = parameters[-2:]
            return [batch_size,input_length,batch_size,output_length]
          
        elif(layer_id == 22):
            
            sigmoid,tanh,ReLU,leaky_ReLU = parameters[-4:]
        
        elif(layer_id == 23):
            return input_size

        elif(layer_id == 24):
            return input_size

        elif(layer_id == 25):
            probability = parameters[-1:][0]

        elif(layer_id == 26):
            probability = parameters[-1:][0]

    def make_layer(self,layer_id,input_size,output_size=None,add_num=None,out_channels=None,target_params=None):
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
        return [1,3,224,224]
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

def make_net_data():
  count = 0
  while(1):
    stream_num = 1
    block_num = random.randint(8,24)
    large = random.randint(0,1)
    try:
      dim = 2
      print(stream_num,block_num,large)
      vg = VectorGenerator(dimension=dim,block_num=block_num,stream_num=stream_num,large = large)
      vg.make_net()
      if not validate_NN(vg,dim):
        stream_num = 1
        block_num = random.randint(8,24)
        continue
      count+=1
      print('计数',count)
    except Exception as e:
      print(e)
      continue

make_net_data()
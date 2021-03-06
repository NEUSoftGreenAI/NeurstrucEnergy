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
import serial #导入模块
np.set_printoptions(threshold=500)


class stableMonitor(Thread):
    def __init__(self, delay):
        super(stableMonitor, self).__init__()
        # print("init monitor...")
        self.stopped = True
        self.drop = False
        self.delay = delay # Time between calls to GPUtil
        self.stable_energy_list = []


        self.ser=serial.Serial('/dev/ttyUSB0',115200,timeout=None)
        self.silence_energy = self.get_silence_energy()
        self.start()

    def get_power(self):
        #十六进制的发送
        hex = b'\x55\x55\x01\x01\x00\x00\xAC' # 对应就是0xef01ffffffff

        self.ser.write(hex)

        #十六进制的读取
        hex_list = []
        
        s = self.ser.read(23).hex()
        for i in range(0,23):
            hex_list.append(s[i*2:i*2+2])
        V = ''.join(hex_list[6:10])
        I = ''.join(hex_list[10:14])
        P = ''.join(hex_list[14:18])
        energy_num = ''.join(hex_list[18:22])
        V = int(V,16) / 1000
        I = int(I,16) / 1000
        P = int(P,16) / 1000
        energy = int(energy_num,16)
        return V,I,P,energy

    def run(self):
      '''
      等待计算完成，完成后等待下一次计算
      '''
      while not self.drop:
        if self.stopped:
          time.sleep(0.001)
        else:
          self.cal_energy()
          
    def stop(self):
      '''
      停止计算功耗
      '''
      self.stopped = True

    def cal_energy(self):
      '''
      获取到达稳定阶段之前的n次推理的开始时间、每隔1ms的能耗、结束时间
      '''
      # print("开始计数")
      energy_list = []
      time_list = []
      screen_energy_list = []
      start_time = int(round(time.time() * 1000))
      while not self.stopped:
            time.sleep(self.delay)
            V,I,P,e = self.get_power()
            energy_list.append(P)
            screen_energy_list.append(e)
            time_list.append(time.time())
      # print("计数结束")
      end_time = int(round(time.time() * 1000))
      forward_energy = 0
      for i in range(1,len(energy_list)):
            forward_energy += (energy_list[i] + energy_list[i])/2 * (time_list[i] - time_list[i-1])
      forward_screen_energy = screen_energy_list[-1] - screen_energy_list[0]
      print("开始时间：%d 结束时间：%d 持续时间：%f秒" % (start_time,end_time,(end_time-start_time)/1000))
      print("计数%d次" % (len(energy_list)))
      print("每隔10次功率：",energy_list[::10])
      print("静默功率：%f W  静默耗能：%f J" % (self.silence_energy,self.silence_energy*(end_time-start_time)/1000))
      print("平均功率：%f W  神经网络耗能：%f J" % (mean(energy_list),(mean(energy_list) - self.silence_energy)*(end_time-start_time)/1000))
      print((mean(energy_list))*(end_time-start_time)/1000,forward_energy,forward_screen_energy)
      length = len(self.stable_energy_list)
      '''
      构造一个长度为10的数组，记录了最近10次的功耗
      '''
      if(length < 3):
        self.stable_energy_list.append((mean(energy_list) - self.silence_energy)*(end_time-start_time)/1000)
      else:
        for i in range(0,2):
          self.stable_energy_list[i] = self.stable_energy_list[i+1]
        self.stable_energy_list[2] = (mean(energy_list) - self.silence_energy)*(end_time-start_time)/1000
      print("表：",self.stable_energy_list)

    def get_silence_energy(self):
      silence_energy_list = []
      for i in range(0,100):
            time.sleep(self.delay)
            V,I,P,e = self.get_power()
            silence_energy_list.append(P)
      return mean(silence_energy_list)

    def begin(self):
      '''
      开始计算功耗
      '''
      time.sleep(1)
      self.silence_energy = self.get_silence_energy()
      time.sleep(2)
      self.stopped = False
    
    def exit(self):
      '''
      线程退出
      '''
      self.drop = True
      self.ser.close()

    def get_stable_energy_list(self):
      return self.stable_energy_list

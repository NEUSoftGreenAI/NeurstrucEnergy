import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import sys
sys.path.append('..') 
from model.BiGNN import BiGNN
from utlis import load_data,load_customdataset_test_data,vaild,get_50_epoch_MAPE,accuracy_train
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--device', type=str, default='1', help='cpu or 0,1,2,3 for gpu')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    if opt.device != 'cpu':
        print(opt.device,type(opt.device))
        torch.cuda.set_device(int(opt.device))
    seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
    path = './CustomModel/'
    _,_,params_min_list,params_max_list,max_layer_length = load_data('custom')
    test_data = load_customdataset_test_data()
    df = pd.DataFrame(columns = ['MobileNetV1','MobileNetV2','ResNet18','InceptionV1','MobileNetV3-SMALL'])
    for model in range(542,542+50):
        checkpoint = torch.load(path +'model_%s' % str(model))
        
        vaild_acc = {
                'MobileNetV1' : [],
                'MobileNetV2' : [],
                'InceptionV1' : [],
                'ResNet18' : [],
                'MobileNetV3-SMALL': []
        }
        gnn = BiGNN(nfeat=110, nhid=80, reverse_hidden=16, nheads=2)
        gnn.load_state_dict(checkpoint['model'])

        gnn.eval()

        for i in range(len(test_data)):
            label = test_data.loc[i]['all_energy']
            pre,MAE,acc = vaild(gnn,params_min_list,params_max_list,max_layer_length,test_data.loc[i]['layer_parameters'],test_data.loc[i]['layer_link'],test_data.loc[i]['layer_id'],label)
            print('%s ,pre: %f ,label: %f, error: %f' %(test_data.loc[i]['name'],pre,test_data.loc[i]['all_energy'],acc.item()))
            vaild_acc[test_data.loc[i]['name']] = acc.item()
        df = df.append(vaild_acc,ignore_index=True)

    print(df)
      

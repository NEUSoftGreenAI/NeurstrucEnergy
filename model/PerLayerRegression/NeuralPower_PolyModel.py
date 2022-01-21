import pandas as pd
import numpy as np
from sklearn import linear_model
import math
# 1000 rondomly selected Conv2d

for polynomial_order in [1,2,3,4]:
    df = pd.read_csv('./data/dataset/energy_conv2d.txt',sep=' ')
    
    data = []
    label = []
    for index,row in df.iterrows():
        if row['data']=='0': 
            continue
        layer_feature = row['data'].split(',')
        layer_feature = [int(x.replace('.0','')) for x in layer_feature]
        weights_num = layer_feature[1] * layer_feature[5] * layer_feature[10] * layer_feature[11]
        OPs = (layer_feature[10] * layer_feature[11] * layer_feature[1] + 1)* layer_feature[6] * layer_feature[7]* layer_feature[5]
        memoryOPs = (layer_feature[1] * layer_feature[2] * layer_feature[3] + layer_feature[5] * layer_feature[6] * layer_feature[7] + weights_num)
        physical_op_params = [weights_num,OPs,memoryOPs]
        data_temp=[]
        for pow_order in range(1,polynomial_order+1):
            layer_feature_poly = [math.pow(x,pow_order) for x in layer_feature]
            physical_op_params_poly = [math.pow(x,pow_order) for x in physical_op_params]
            data_temp.extend(layer_feature_poly + physical_op_params_poly)
        # print(data_temp)
        data.append(data_temp)
        label.append(row['all_energy'])

    random_mape_record = []

    for alpha in np.arange(0, 0.002, 0.0001):
        
        reg=linear_model.Lasso(alpha=alpha,max_iter=100000, normalize=True).fit(data,label)
        predict = reg.predict(data)
        acc = []
        error = abs(predict-label) / label
        mape = np.mean(error)
        # print('Random',alpha,mape)
        random_mape_record.append(mape)
    
    # 60 Conv2d from PreVIous

    df = pd.read_csv('./data/dataset/energy_conv2d_PreVIousNet.txt',sep=' ')
    data = []
    label = []
    
    for index,row in df.iterrows():
        if row['data']=='0': 
            continue

        layer_feature = row['data'].split(',')
        layer_feature = [int(x.replace('.0','')) for x in layer_feature]
        weights_num = layer_feature[1] * layer_feature[5] * layer_feature[10] * layer_feature[11]
        OPs = (layer_feature[10] * layer_feature[11] * layer_feature[1] + 1)* layer_feature[6] * layer_feature[7]* layer_feature[5]
        memoryOPs = (layer_feature[1] * layer_feature[2] * layer_feature[3] + layer_feature[5] * layer_feature[6] * layer_feature[7] + weights_num)
        physical_op_params = [weights_num,OPs,memoryOPs]
        data_temp=[]
        for pow_order in range(1,polynomial_order+1):
            layer_feature_poly = [math.pow(x,pow_order) for x in layer_feature]
            physical_op_params_poly = [math.pow(x,pow_order) for x in physical_op_params]
            data_temp.extend(layer_feature_poly + physical_op_params_poly)
        data.append(data_temp)
        label.append(row['all_energy'])

    custom_mape_record = []

    for alpha in np.arange(0, 0.002, 0.0001):
        reg=linear_model.Lasso(alpha=alpha,max_iter=100000, normalize=True).fit(data,label)
        predict = reg.predict(data)
        acc = []
        error = abs(predict-label) / label
        mape = np.mean(error)   
        # print('Random',alpha,mape)
        custom_mape_record.append(mape)
    print('NeuralPower: Conv-R MAPE = {:.2%}  Conv-P MAPE = {:.2%} with polynomial = {}'.format(min(random_mape_record),min(custom_mape_record),polynomial_order))

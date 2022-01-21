import pandas as pd
import numpy as np
from sklearn import linear_model

# 1000 rondomly selected Conv2d

df = pd.read_csv('./data/dataset/energy_conv2d.txt',sep=' ')
data = []
label = []
for index,row in df.iterrows():
    if row['data']=='0': 
        continue

    params = row['data'].split(',')
    params = [int(x.replace('.0','')) for x in params]
    weights_num = params[1] * params[5] * params[10] * params[11]
    OPs = (params[10] * params[11] * params[1] + 1)* params[6] * params[7]* params[5] 
    memoryOPs = (params[1] * params[2] * params[3] + params[5] * params[6] * params[7] + weights_num)
    data.append([weights_num,OPs,memoryOPs])
    label.append(row['all_energy'])

random_mape_record = []

for alpha in np.arange(0, 0.02, 0.0001):
    reg=linear_model.Ridge(alpha=alpha, max_iter=100000, normalize=True).fit(data,label)
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

    params = row['data'].split(',')
    params = [int(x.replace('.0','')) for x in params]
    weights_num = params[1] * params[5] * params[10] * params[11]
    OPs = (params[10] * params[11] * params[1] + 1)* params[6] * params[7]* params[5] 
    memoryOPs = (params[1] * params[2] * params[3] + params[5] * params[6] * params[7] + weights_num)
    data.append([weights_num,OPs,memoryOPs])
    label.append(row['all_energy'])

custom_mape_record = []

for alpha in np.arange(0, 0.02, 0.0001):
    reg=linear_model.Ridge(alpha=alpha, max_iter=100000, normalize=True).fit(data,label)
    predict = reg.predict(data)
    acc = []
    error = abs(predict-label) / label
    mape = np.mean(error)
    # print('Random',alpha,mape)
    custom_mape_record.append(mape)
print('PreVIous: Conv-R MAPE = {:.2%}  Conv-P MAPE = {:.2%}'.format(min(random_mape_record),min(custom_mape_record)))

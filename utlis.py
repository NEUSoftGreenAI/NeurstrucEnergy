
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data


def get_params_length(layer_id):
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

def link_vector_to_graph(link_list,length,max_layer_length):
    '''
    将连接向量转化成邻接矩阵，对角线元素表示是否接收初始输入
    '''
    adj = np.zeros((max_layer_length,max_layer_length))
    
    graph = np.zeros([length,length],dtype = float)
    flag = 0
    # print(link_list,length,max_layer_length)
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
                
    adj[0:length,0:length] = graph
    for i in range(length):
      adj[i][i] = 1
    return adj.T

def get_params_position(id):
    params_length_dic = {
        0:0,
        1:19,
        2:0,
        3:0,
        4:20,
        5:0,
        6:0,
        7:17,
        8:0,
        9:0,
        10:14,
        11:0,
        12:0,
        13:9,
        14:0,
        15:0,
        16:5,
        17:0,
        18:0,
        19:5,
        20:0,
        21:4,
        22:6,
        23:3,
        24:3,
        25:5,
        26:0
    }
    start = 0
    end = 0
    for i in range(26):
      if i != id:
        start += params_length_dic[i]
        end += params_length_dic[i]
      else:
        end += params_length_dic[i]
        break
    return start,end

def load_randomdataset_test_data():
    df_1 = pd.read_csv('../data/dataset/random_testset_1.txt',sep = ' ',index_col=False)
    df_2 = pd.read_csv('../data/dataset/random_testset_2.txt',sep = ' ',index_col=False)
    df_3 = pd.read_csv('../data/dataset/random_testset_3.txt',sep = ' ',index_col=False)
    mean_energy =(df_1['all_energy'] + df_2['all_energy'] + df_3['all_energy']) / 3
    df_1['all_energy'] = mean_energy
    
    return df_1

def load_customdataset_test_data():
    df_1 = pd.read_csv('../data/dataset/custom_testset_1.txt',sep = ' ',index_col=False)
    df_2 = pd.read_csv('../data/dataset/custom_testset_2.txt',sep = ' ',index_col=False)
    df_3 = pd.read_csv('../data/dataset/custom_testset_3.txt',sep = ' ',index_col=False)
    mean_energy =(df_1['all_energy'] + df_2['all_energy'] + df_3['all_energy']) / 3
    df_1['all_energy'] = mean_energy
    
    return df_1

def vaild(model,params_min_list,params_max_list,max_layer_length,layer_parameters,layer_link,layer_id,energy,split_gap = 24,split_index_list = None):
    layer_parameters = np.array([float(x) if '.' in x else int(x) for x in layer_parameters.split(',')],dtype='float')
    layer_link = np.array([int(x.replace('.0','')) for x in layer_link.split(',')])
    layer_id = np.array([int(x) for x in layer_id.split(',')])
    # array = np.zeros(1)
    energy = [energy]
    index = 0
    for id in layer_id:

        params_length = get_params_length(id)
        params = layer_parameters[index:index+params_length]
        params = [(params[j] - params_min_list[id][j]) / (params_max_list[id][j]) if params_max_list[id][j] != 0 or params_min_list[id][j] != params_max_list[id][j] else 0 for j in range(params_length)]
        layer_parameters[index:index+params_length] = params
        index += params_length
        
    index = 0
    layer_params = []
    for id in layer_id:
        params = [0 for i in range(110)]
        start,end = get_params_position(id)
        params_length = get_params_length(id)
        params[start:end] = layer_parameters[index:index + params_length].tolist()
        layer_params.append(params)
        index += params_length
    
    adj = link_vector_to_graph(layer_link,len(layer_id),max_layer_length)

    layer_id = layer_id.tolist()
    if len(layer_id) < max_layer_length:
        for j in range(0,max_layer_length - len(layer_id)):
            layer_params.append([0 for i in range(110)])
        layer_id.extend([-1 for i in range(max_layer_length - len(layer_id))]) #层数量长度不足的填充-1
        
        adj = torch.ShortTensor(np.array(adj)).unsqueeze(0).cuda() # [1,70,294]
        data_x = torch.FloatTensor(np.array(layer_params)).unsqueeze(0).cuda() # [1,70,294]
        data_id = np.array(layer_id)
        data_id = torch.FloatTensor(data_id).unsqueeze(0).cuda()
        # print()
        output = model(data_x, adj, data_id)
        # output = torch.squeeze(output, dim=0)
        # print(output)
        MAE_error = abs(output.item() - energy[0])
        error_val = accuracy_test(output.cpu(),energy[0])
        
        return output, MAE_error, error_val

def accuracy_test(output, labels):
    
    return abs(output - labels)/labels * 100

def load_data(dataset_type):
    
    print('load data...')
    #存储每类层的所有元素，方便后续计算最大值最小值
    params_list = {
        0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[],21:[],22:[],23:[],24:[],25:[],26:[]
    }
    #存储每类层，各个元素的最小值
    params_min_list = {
        0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[],21:[],22:[],23:[],24:[],25:[],26:[]
    }
    #存储每类层，各个元素的最小值
    params_max_list = {
        0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[],21:[],22:[],23:[],24:[],25:[],26:[]
    }
    data = pd.read_csv('../data/dataset/%s_data.txt' % dataset_type,sep = ' ',index_col=False)

    layer_parameters = data['layer_parameters'].values
    layer_link = data['layer_link'].values
    layer_id = data['layer_id'].values

    max_layer_length = max([len(layer_id.split(',')) for layer_id in data['layer_id'].values]) #获取最长的层数
    # print(max_layer_length)
    for i in range(len(layer_parameters)):
        try:
            layer_parameters[i] = np.array([float(x) if '.' in x else int(x) for x in layer_parameters[i].split(',')],dtype='float')
            layer_link[i] = np.array([int(x) for x in layer_link[i].split(',')])
            layer_id[i] = np.array([int(x) for x in layer_id[i].split(',')])
        except:
            print(i,layer_parameters[i],layer_id[i])
          

    for i in range(len(layer_parameters)):
        one_net_layer_id = layer_id[i]
        index = 0
        for id in one_net_layer_id:
            params_length = get_params_length(id)
            params = layer_parameters[i][index:index+params_length]
            index += params_length
            params_list[id].append(params.tolist())

    for i in range(0,27):
        if len(params_list[i]) != 0:
            params_max_list[i] = np.amax(np.array(params_list[i]), axis=0)
            params_min_list[i] = np.amin(np.array(params_list[i]), axis=0)
    
    # 归一化
    for i in range(len(layer_parameters)):
        one_net_layer_id = layer_id[i]
        index = 0
        #对不同层，分别归一化
        for id in one_net_layer_id:
            params_length = get_params_length(id)
            params = layer_parameters[i][index:index+params_length]
            params = [(params[j] - params_min_list[id][j]) / (params_max_list[id][j]) if params_max_list[id][j] != 0 else 0 for j in range(params_length)]
            layer_parameters[i][index:index+params_length] = params
            index += params_length

    all_params_array = []
    all_id_array = []
    all_adj_array = []
    data_link_all = torch.IntTensor()
    for i in range(0,len(layer_parameters)):
        # if i % 1000 == 0 and i == 1000:
        #   data_link = torch.IntTensor(np.array(all_adj_array))
        #   data_link_all = data_link
        #   all_adj_array = []
        if i % 1000 == 0 and i != 0:
            data_link = torch.IntTensor(np.array(all_adj_array))
            data_link_all = torch.cat([data_link_all,data_link])
            all_adj_array = []

        net_adj = link_vector_to_graph(layer_link[i],len(layer_id[i]),max_layer_length)
        all_adj_array.append(net_adj)
    # print(all_adj_array[0])
    data_link = torch.IntTensor(np.array(all_adj_array))
    data_link_all = torch.cat([data_link_all,data_link])
    print(data_link_all.shape)
    for i in range(0,len(layer_parameters)):
        index = 0
        layer_params = []
        for id in layer_id[i]:
            params = [0 for i in range(110)]
            start,end = get_params_position(id)
            params_length = get_params_length(id)
            if id != 23 or id != 24:
                params[start:end] = layer_parameters[i][index:index + params_length].tolist()
                layer_params.append(params)
            index += params_length

        for j in range(0,max_layer_length - len(layer_id[i])):
            layer_params.append([0 for i in range(110)])

        for j in range(len(layer_id[i])):
            id = layer_id[i][j]
            if id == 23 or id == 24:
                for k in range(j,len(layer_id[i])-1):
                    layer_id[i][k] = layer_id[i][k+1]
                layer_id[i][len(layer_id[i])-1] = -1
            

        layer_id[i] = layer_id[i].tolist()
        layer_id[i].extend([-1 for i in range(max_layer_length - len(layer_id[i]))]) #层数量长度不足的填充-1
        all_id_array.append(layer_id[i])
        all_params_array.append(layer_params)
        
    # b = np.load("all_params_array.npy")
    # data_link = torch.FloatTensor(np.array(all_adj_array))
    data_x = torch.FloatTensor(np.array(all_params_array))

    data_id = np.array(all_id_array)
    data_id = torch.FloatTensor(data_id)
    
    data_y = torch.FloatTensor(data['all_energy'].values)

    train_size = int(0.8 * len(data_x))
    test_size = len(data_x) - train_size
    BATCH_SIZE = 128
    full_dataset = Data.TensorDataset(data_x, data_id, data_link_all, data_y) #将x,y读取，转换成Tensor格式
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # 最新批数据
        shuffle=True,               # 是否随机打乱数据
        num_workers=0,              # 用于加载数据的子进程
    )

    # test_torch_dataset = Data.TensorDataset(test_params_inputs, test_id_inputs, test_outputs) #将x,y读取，转换成Tensor格式

    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # 最新批数据
        shuffle=True,               # 是否随机打乱数据
        num_workers=0,              # 用于加载数据的子进程
    )

    return train_loader,test_loader,params_min_list,params_max_list,max_layer_length

def get_50_epoch_MAPE(epoch,vaild_acc):
    all_test_mean = 0
    all_test_mean_list = []
    count = 0
    if epoch < 50:
        start_index = 0
    else:
        start_index = epoch - 50
    for net_name,acc_list in vaild_acc.items():
        count += 1
        all_test_mean += np.mean(acc_list[start_index:epoch],axis=0)[0]
        all_test_mean_list.append(np.mean(acc_list[start_index:epoch],axis=0)[0])
    all_test_mean_list.sort()
    return np.mean(all_test_mean_list[0:18])

def accuracy_train(output, labels):
    
    output = output.cpu().detach().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    for i in range(0,len(output)):
        output[i] = abs(output[i] - labels[i])/labels[i] * 100
    return np.mean(output)
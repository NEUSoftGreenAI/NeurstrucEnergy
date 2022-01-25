from BiGNN import BiGNN
import sys
sys.path.append('..') 
from utlis import load_data,load_customdataset_test_data,load_randomdataset_test_data,vaild,get_50_epoch_MAPE,accuracy_train
import argparse
import random
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import os
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--weight-decay', type=float, default=4e-5)
    parser.add_argument('--hidden', type=int, default=96, help='hidden size of forward aggregator')
    parser.add_argument('--reverse-hidden', type=int, default=16, help='hidden size of reverse aggregator')
    parser.add_argument('--lr', type=float, default=0.0006, help='learning rate')
    parser.add_argument('--heads', type=int, default=2, help='number of heads')
    parser.add_argument('--dataset', type=str, default='random', help='random or custom')
    parser.add_argument('--step', type=int, default=30, help='step of lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.75, help='decay ratio of lr scheduler')
    parser.add_argument('--device', type=str, default='1', help='cpu or 0,1,2,3 for gpu')
    
    opt = parser.parse_args()
    return opt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
    
def save_model(model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, '/content/drive/MyDrive/RandomModel/model_%s' % epoch)

if __name__ == '__main__':
    opt = parse_opt()
    set_seed(opt.seed)
    if opt.device != 'cpu':
        print(opt.device,type(opt.device))
        torch.cuda.set_device(int(opt.device))
    train_loader, vaild_loader, params_min_list, params_max_list, max_layer_length = load_data(opt.dataset)
    if opt.dataset == 'random':
        test_data = load_randomdataset_test_data()
    else:
        test_data = load_customdataset_test_data()
    
    
    min_loss = 1e10
    test_net_name = test_data['name'].values.tolist()
    vaild_acc = {}
    for name in test_net_name:
        vaild_acc[name] = []

    # Model and optimizer
    model = BiGNN(nfeat = 110,
            nhid=opt.hidden,
            reverse_hidden=opt.reverse_hidden,
            nheads=opt.heads)
    optimizer = optim.Adam(model.parameters(), 
            lr=opt.lr, 
            weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    cuda = torch.cuda.is_available()
    if cuda:
        print('use GPU')
        model.cuda()

    loss_func = torch.nn.L1Loss()

    # Train model
    epochs = opt.epochs
    t_total = time.time()
    loss_values = []
    error_values = []
    test_error_list = []
    vaild_loss_values = []
    vaild_error_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0
    patience_cnt = 0
    patience = 200
    for epoch in range(epochs):
        t = time.time()
        step_loss_values = []
        step_error_values = []
        scheduler.step()
        for step, (batch_params, batch_id, batch_link, batch_y) in enumerate(train_loader):  # 每个训练步骤
            if cuda:
                batch_params = batch_params.cuda()
                batch_id = batch_id.cuda()
                batch_y = batch_y.cuda()
                batch_link = batch_link.cuda()

            model.train()
            optimizer.zero_grad()
            output = model(batch_params, batch_link, batch_id)
            loss_train = loss_func(output, batch_y)
            error_train = accuracy_train(output, batch_y)
            loss_train.backward()
            optimizer.step()
            step_loss_values.append(loss_train.item())
            step_error_values.append(error_train)
            
            
        # print(step_loss_values)   
        loss_values.append(np.mean(step_loss_values))
        error_values.append(np.mean(step_error_values))
        epoch_train_loss = np.mean(step_loss_values)
        epoch_train_error = np.mean(step_error_values)
        step_loss_values = []
        step_error_values = []
        
        for step, (batch_params, batch_id, batch_link, batch_y) in enumerate(vaild_loader):  # 每个训练步骤
            if cuda:
                batch_params = batch_params.cuda()
                batch_id = batch_id.cuda()
                batch_y = batch_y.cuda()
                batch_link = batch_link.cuda()

            model.eval()
            output = model(batch_params, batch_link, batch_id)
            loss_val = loss_func(output, batch_y)
            error_val = accuracy_train(output, batch_y)
            step_loss_values.append(loss_val.item())
            step_error_values.append(error_val)
            
        epoch_val_loss = np.mean(step_loss_values)
        epoch_val_error = np.mean(step_error_values)
        vaild_loss_values.append(epoch_val_loss)
        vaild_error_values.append(epoch_val_error)
        
        #测试
        vaild_mean = 0
        count = 0 
        for i in range(len(test_data)):
            label = test_data.loc[i]['all_energy']
            pre,MAE,acc = vaild(model,params_min_list,params_max_list,max_layer_length,test_data.loc[i]['layer_parameters'],test_data.loc[i]['layer_link'],test_data.loc[i]['layer_id'],label)
            # print('%s ,pre: %f ,label: %f, error: %f' %(test_data.loc[i]['name'],pre,test_data.loc[i]['all_energy'],acc.item()))
              
            vaild_mean += acc.item()
            count += 1
            vaild_acc[test_data.loc[i]['name']].append([acc.item(),pre.item()])
        vaild_mean = vaild_mean / count
        if epoch > 1:
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(epoch_train_loss),
                'error_train: {:.4f}'.format(epoch_train_error),
                'loss_val: {:.4f}'.format(epoch_val_loss),
                'error_val: {:.4f}'.format(epoch_val_error),
                'time: {:.4f}s'.format(time.time() - t),
                'error_test: {:.4f}'.format(vaild_mean),
                '100_epoch_test: {:.4f}'.format(get_50_epoch_MAPE(epoch,vaild_acc)))
            test_error_list.append(vaild_mean)
        
            if test_error_list[-1] < min_loss:
                min_loss = test_error_list[-1]
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1
          
            if patience_cnt == patience:
                break

            if epoch == 100 and get_50_epoch_MAPE(epoch,vaild_acc) > 18:
                break
          # if epoch >= 243-50 and epoch <= 243:
          #   print('save')
          #   save_model(model, optimizer, epoch)
        
        all_test_mean = 0
        if epoch>50 and epoch %10 == 1:
            index = epoch - 50
            split = epoch
            print('\n',index,split,'-----------------','\n')
            print('Test Acc',np.mean(vaild_error_values[index:split]))
            for net_name,acc_list in vaild_acc.items():
                # all_test_mean += np.mean(acc_list[index:split])
                print(net_name,np.mean(acc_list[index:split],axis=0)[0],np.mean(acc_list[index:split],axis=0)[1])
            index = split
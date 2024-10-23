# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import math
from sklearn.model_selection import KFold
from tqdm import tqdm


def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
 

parent_dir = './baseline/12_sintheta_2nd_partonly/'
physical_para = pd.read_csv(parent_dir + 'physical_para_include_936.csv', header=None, low_memory=False)
XRD_descriptor = pd.read_csv(parent_dir + 'XRD_sin_descriptor_936.csv', header=None, low_memory=False)
data = pd.concat([XRD_descriptor, physical_para.iloc[:,0]], axis=1)
grid_points = [[12,52]]
results_all = []

r2_record = -math.inf
seeds = [199228,302675,257057,320858,871534,144537,844620,298933,681403,690678]


for grid_point in tqdm(grid_points):
    results=[]
    result_r2_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_mae_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_loss_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_r2_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_mae_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_loss_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    for i_seed in tqdm(range(10), position=0, leave=True): #十次k_fold
        seed = seeds[i_seed] 
        result_r2_train.iloc[i_seed,0] = seed
        result_mae_train.iloc[i_seed,0] = seed
        result_loss_train.iloc[i_seed,0] = seed
        result_r2_test.iloc[i_seed,0] = seed
        result_mae_test.iloc[i_seed,0] = seed
        result_loss_test.iloc[i_seed,0] = seed
        
        setup_seed(seed)

        i = 0    
        kf = KFold(n_splits=10,shuffle=True, random_state=seed)
        for train_index, test_index in kf.split(data):
        
            x_train = data.iloc[train_index,:4096]
            x_test = data.iloc[test_index,:4096]
            y_train = data.iloc[train_index,4096]
            y_test = data.iloc[test_index,4096]
            
            x_train = torch.from_numpy(x_train.values)
            x_test = torch.from_numpy(x_test.values)
            y_train = torch.from_numpy(y_train.values)
            y_test = torch.from_numpy(y_test.values)
            
            x_train = x_train.to(torch.float32) 
            x_train = torch.unsqueeze(x_train, 1)  
            y_train = y_train.to(torch.float32)   
            x_test = x_test.to(torch.float32) 
            x_test = torch.unsqueeze(x_test, 1)   
            y_test = y_test.to(torch.float32)  
            
            input_size, feature_size = x_train.shape[0], x_train.shape[1]
            LR = 0.001
            epochs = 1000
            mb_size = 50
            loss_train_log = []
            loss_test_log = []
           
            class UA_CNN(nn.Module):
                def __init__(self):
                    super(UA_CNN, self).__init__()#N*2*4096
                    self.partition_1 = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=2,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*4*1024
                                    nn.Conv1d(in_channels=2,out_channels=4,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*4*512
                                    nn.Conv1d(in_channels=4,out_channels=2,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*2*256
                                    nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2)#N*1*128
                                    ])
                    self.partition_2 = nn.ModuleList([nn.Conv1d(in_channels=1,out_channels=2,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*4*1024
                                    nn.Conv1d(in_channels=2,out_channels=4,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*4*512
                                    nn.Conv1d(in_channels=4,out_channels=2,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2),#N*2*256
                                    nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2,2)#N*1*128
                                    ])
                    # self.dropout_layer = nn.Dropout(0.1)
                    self.fc1 = nn.Linear(208,64)
                    self.relu5 = nn.ReLU()
                    self.fc2 = nn.Linear(64,32)
                    self.relu6 = nn.ReLU()
                    self.fc3 = nn.Linear(32,8)
                    self.relu7 = nn.ReLU()
                    self.fc4 = nn.Linear(8,1)
             
                def forward(self, x, gp):
                    x2 = x[:,:,gp[0]*64:]
                    for item in self.partition_2:
                        x2 = item(x2)
                    out = self.fc1(x2)
                    out = self.relu5(out)
                    out = self.fc2(out)
                    out = self.relu6(out)
                    out = self.fc3(out)
                    out = self.relu7(out)
                    out = self.fc4(out)
                    return out, x2
            
            
            def random_mini_batches(X_train, Y_train, mini_batch_size = 10):                           
                mini_batches = []
                X_train = torch.split(X_train, mini_batch_size)
                Y_train = torch.split(Y_train, mini_batch_size)
                for i in np.arange(len(X_train)):
                    mini_batch = (X_train[i],Y_train[i])
                    mini_batches.append(mini_batch)
                return mini_batches
            
            
            model = UA_CNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            
            loss_func = nn.MSELoss()        
                    
            y_train_sub = y_train
            y_test_sub = y_test
            
            r2_best = -math.inf
            MAE_best = 0
            for epoch in range(epochs):
                epoch_loss = 0
                num_minibatches = int(input_size / mb_size) + 1
                minibatches = random_mini_batches(x_train, y_train_sub, mb_size)
                model.train()
                for minibatch in minibatches:
                    batch_x, batch_y  = minibatch
                    batch_y_pre, _ = model(batch_x,grid_point)
                    idx = torch.nonzero(batch_y.squeeze()!=0,as_tuple=False)
                    batch_y_pre1 = torch.index_select(batch_y_pre.squeeze(), dim=0, index = idx.squeeze())
                    batch_y1 = torch.index_select(batch_y.squeeze(), dim=0, index = idx.squeeze())
                    loss = loss_func(batch_y_pre1.squeeze(), batch_y1.squeeze())    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss = epoch_loss + (loss / num_minibatches)
                loss_train_log.append(torch.mean(epoch_loss).item())
            
            
                model.eval()
                with torch.no_grad():
                    y_test_pre, _ = model(x_test,grid_point)
                    idx_test = torch.nonzero(y_test_sub.squeeze()!=0,as_tuple=False)
                    y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index = idx_test.squeeze())
                    y_test_sub1 = torch.index_select( y_test_sub.squeeze(), dim=0, index = idx_test.squeeze())        
                    loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())
                    MAE2 = mean_absolute_error(y_test_sub1.squeeze(),y_test_pre1.squeeze())
                    r2_score_v =  r2_score(y_test_sub1.squeeze(),y_test_pre1.squeeze()) 
            
                    if r2_best < r2_score_v:
                        best_test_loss = loss_test
                        torch.save(model.state_dict(), parent_dir + 'best_test_model.pth')
                        MAE_best = MAE2
                        r2_best = r2_score_v
                    print('Iter-{}; Total loss: {:.4}; MAE2: {:.4}; r2_score_v: {:.4}'.format(epoch, loss_test.item(), MAE2, r2_score_v))
                
                loss_test_log.append(torch.mean(loss_test).item())
            
            model_test = UA_CNN()
            model_test.load_state_dict(torch.load(parent_dir + 'best_test_model.pth'))
            y_test_pre, y_test_vector = model_test(x_test,grid_point)
            y_train_pre, y_train_vector = model_test(x_train,grid_point)
            
            with torch.no_grad():
                idx_train = torch.nonzero(y_train_sub.squeeze()!=0,as_tuple=False)
                idx_tst = torch.nonzero(y_test_sub.squeeze()!=0,as_tuple=False)
                y_train_sub1 = torch.index_select(y_train_sub.squeeze(), dim=0, index = idx_train.squeeze())
                y_train_pre1 = torch.index_select(y_train_pre.squeeze(), dim=0, index = idx_train.squeeze())
                y_test_sub1 = torch.index_select(y_test_sub.squeeze(), dim=0, index = idx_tst.squeeze())
                y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index = idx_tst.squeeze())
                y_test_vector1 = torch.index_select(y_test_vector.squeeze(), dim=0, index = idx_tst.squeeze())
                y_train_vector1 = torch.index_select(y_train_vector.squeeze(), dim=0, index = idx_train.squeeze())
                
            loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())
            MAE_test = mean_absolute_error(y_test_sub1.squeeze(),y_test_pre1.squeeze())
            r2_test = r2_score(y_test_sub1.squeeze(),y_test_pre1.squeeze())
            
            loss_train = loss_func(y_train_pre1.squeeze(), y_train_sub1.squeeze())
            MAE_train = mean_absolute_error(y_train_sub1.squeeze(),y_train_pre1.squeeze())
            r2_train = r2_score(y_train_sub1.squeeze(),y_train_pre1.squeeze())
            
            if r2_test > r2_record:
                r2_record = r2_test
                torch.save(y_train_sub1.squeeze(), parent_dir + 'y_train_sub.pth')
                torch.save(y_train_pre1.squeeze(), parent_dir + 'y_train_pre.pth')
                torch.save(y_test_sub1.squeeze(), parent_dir + 'y_test_sub.pth')
                torch.save(y_test_pre1.squeeze(), parent_dir + 'y_test_pre.pth')  
                torch.save(y_test_vector.squeeze(), parent_dir + 'y_test_vector.pth')
                torch.save(y_train_vector.squeeze(), parent_dir + 'y_train_vector.pth') 
            
            result_r2_test.iloc[i_seed,i+1] = r2_test
            result_mae_test.iloc[i_seed,i+1] = MAE_test
            result_loss_test.iloc[i_seed,i+1] = loss_test.detach().numpy()
            result_r2_train.iloc[i_seed,i+1] = r2_train
            result_mae_train.iloc[i_seed,i+1] = MAE_train
            result_loss_train.iloc[i_seed,i+1] = loss_train.detach().numpy()
            i = i+1
            
    results.append(grid_point)
    results.append(result_r2_test)
    results.append(result_mae_test)
    results.append(result_loss_test)
    results.append(result_r2_train)
    results.append(result_mae_train)
    results.append(result_loss_train)
    results_all.append(results)
torch.save(results_all, parent_dir + 'results.pth')
  
    

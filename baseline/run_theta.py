import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import math
from utils import *
import numpy as np
import pandas as pd
import os
import pandas as pd
import os

import pandas as pd
import os

def save_results_to_excel(file_path, experiment_num, avg_preds, epi_std, ale_std, total_std, total_std_2, y):
    
    avg_preds_flat = [item for sublist in avg_preds for item in sublist]
    epi_std_flat = epi_std.tolist()
    ale_std_flat = ale_std.tolist()
    total_std_flat = total_std.tolist()
    total_std_2_flat = total_std_2.tolist()
    y_flat = y.tolist()  

    
    data_dict = {
        'Experiment': [experiment_num] * len(avg_preds_flat),
        'y': y_flat,  
        'avg_preds': avg_preds_flat,
        'epi_std': epi_std_flat,
        'ale_std': ale_std_flat,
        'total_std': total_std_flat,
        'total_std_2': total_std_2_flat
    }
    
    df = pd.DataFrame(data_dict)

    print(f"Saving results for Experiment {experiment_num} to {file_path}")  
    print(f"DataFrame to save: {df.head()}")  

    
    if not os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'Experiment_{experiment_num}', index=False)
    else:
        
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            try:
                df.to_excel(writer, sheet_name=f'Experiment_{experiment_num}', index=False)
            except KeyError:
                df.to_excel(writer, sheet_name=f'Experiment_{experiment_num}', index=False)


class UA_CNN(nn.Module):
    def __init__(self):
        super(UA_CNN, self).__init__()#N*2*4096
        
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=2,kernel_size=3,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2,2)#N*4*2048
        self.conv2 = nn.Conv1d(in_channels=2,out_channels=4,kernel_size=3,padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2,2)#N*4*1024
        self.conv3 = nn.Conv1d(in_channels=4,out_channels=2,kernel_size=3,padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(2,2)#N*2*512
        self.conv4 = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool1d(2,2)#N*1*256
#        self.dropout_layer = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256,64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64,32)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(32,16)
        self.relu7 = nn.ReLU()
        self.fc4 = nn.Linear(16,8)
        self.relu8 = nn.ReLU()
        self.fc5 = nn.Linear(8,1)
        
        self.fc1_var = nn.Linear(256,64)
        self.relu5_var = nn.ReLU()
        self.fc2_var = nn.Linear(64,32)
        self.relu6_var = nn.ReLU()
        self.fc3_var = nn.Linear(32,16)
        self.relu7_var = nn.ReLU()
        self.fc4_var = nn.Linear(16,8)
        self.relu8_var = nn.ReLU()
        self.fc5_var = nn.Linear(8,1)
    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = self.maxpool3(self.relu3(self.conv3(out)))
        out_vector = self.maxpool4(self.relu4(self.conv4(out)))
#        out = self.dropout_layer(out)
        out = self.fc1(out_vector)
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        out = self.relu7(out)
        out = self.fc4(out)
        out = self.relu8(out)
        out = self.fc5(out)
        
        out_var = self.fc1_var(out_vector)
        out_var = self.relu5_var(out_var)
        out_var = self.fc2_var(out_var)
        out_var = self.relu6_var(out_var)
        out_var = self.fc3_var(out_var)
        out_var = self.relu7_var(out_var)
        out_var = self.fc4_var(out_var)
        out_var = self.relu8_var(out_var)
        out_var = self.fc5_var(out_var)
        
        return out, out_vector, out_var
    
import os
import datetime
parent_dir = './baseline/'
# Set up the new results directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(parent_dir, f'results_new/m1_att0/{timestamp}')
# 设置结果Excel的文件路径
results_excel_file = os.path.join(results_dir, 'm1_att0_uncertainty_results.xlsx')

os.makedirs(results_dir, exist_ok=True)
# Create a log file with a timestamp
log_file_path = os.path.join(results_dir, f'training_log_{timestamp}.txt')
# Function to log and print messages
def log_and_print(message, log_file_path):
    print(message)
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

physical_para = pd.read_csv('../data/934label-16_no-header.csv', header=None, low_memory=False)
XRD_descriptor = pd.read_csv('../data/XRD_descriptor_936.csv', header=None, low_memory=False)
data = pd.concat([XRD_descriptor, physical_para.iloc[:,0]], axis=1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# heteroscedastic_loss_coefficient = 1e-3
seeds = range(1,100)
seeds_len = 5
num_seeds = len(seeds)

total_experiments = 10

for exp_num in range(total_experiments):
    print(f"Starting experiment {exp_num + 1}")  
    
    start_index = exp_num % num_seeds  
    selected_seeds = [seeds[(start_index + i) % num_seeds] for i in range(seeds_len)]
    print('selected_seeds', selected_seeds)
    
    heteroscedastic_loss_coefficient = 1e-3
    results_all = []
    LR = 0.001
    epochs = 1000
    mb_size = 50
    r2_record = -math.inf

    indices = data.index
    
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    x_train = data.iloc[train_indices, :4096]
    x_test = data.iloc[test_indices, :4096]
    y_train = data.iloc[train_indices, 4096]
    y_test = data.iloc[test_indices, 4096]
    
    y_test_ = torch.tensor(y_test.values)  
    idx_test = torch.nonzero(y_test_.squeeze() != 0, as_tuple=False)
    x_test_len = len(idx_test)
    log_and_print(f"x_test_len: {x_test_len}", log_file_path)

    x_train = torch.from_numpy(x_train.values).float().to(device)
    x_test = torch.from_numpy(x_test.values).float().to(device)
    y_train = torch.from_numpy(y_train.values).float().to(device)
    y_test = torch.from_numpy(y_test.values).float().to(device)

    x_train = torch.unsqueeze(x_train, 1)
    x_test = torch.unsqueeze(x_test, 1)

    sum_preds = np.zeros((x_test_len, 1))
    sum_ale_uncs = np.zeros((x_test_len, 1))
    sum_epi_uncs = np.zeros((x_test_len, 1))
    all_preds = np.zeros((x_test_len, 1, seeds_len))

    results=[]
    result_r2_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_mae_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_loss_test = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_r2_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_mae_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    result_loss_train = pd.DataFrame(np.zeros([10,11]),columns=['seed','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
    
    for i_seed in tqdm(range(len(selected_seeds)), position=0, leave=True): # Use seeds_len to match the initialization
        seed = selected_seeds[i_seed]
        result_r2_train.iloc[i_seed,0] = seed
        result_mae_train.iloc[i_seed,0] = seed
        result_loss_train.iloc[i_seed,0] = seed
        result_r2_test.iloc[i_seed,0] = seed
        result_mae_test.iloc[i_seed,0] = seed
        result_loss_test.iloc[i_seed,0] = seed
        setup_seed(seed)
        i = 0    
        
        input_size, feature_size = x_train.shape[0], x_train.shape[1]
        loss_train_log = []
        loss_test_log = []
        
        model = UA_CNN().to(device)
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
                batch_y_pre, _, batch_y_pre_log_var = model(batch_x)
                idx = torch.nonzero(batch_y.squeeze()!=0,as_tuple=False)
                batch_y_pre1 = torch.index_select(batch_y_pre.squeeze(), dim=0, index = idx.squeeze())
                batch_y_pre_log_var1 = torch.index_select(batch_y_pre_log_var.squeeze(), dim=0, index = idx.squeeze())
                batch_y1 = torch.index_select(batch_y.squeeze(), dim=0, index = idx.squeeze())
                mse_loss = loss_func(batch_y_pre1.squeeze(), batch_y1.squeeze())    
                h_loss = heteroscedastic_loss(batch_y1.squeeze(),batch_y_pre1.squeeze(),batch_y_pre_log_var1.squeeze())
                loss = mse_loss + heteroscedastic_loss_coefficient * h_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss = epoch_loss + (loss / num_minibatches)
            loss_train_log.append(torch.mean(epoch_loss).item())
        
            model.eval()
            with torch.no_grad():
                y_test_pre, _, y_test_pre_log_var = model(x_test)
                idx_test = torch.nonzero(y_test_sub.squeeze()!=0,as_tuple=False)
                y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index = idx_test.squeeze())
                y_test_pre_log_var1 = torch.index_select(y_test_pre_log_var.squeeze(), dim=0, index = idx_test.squeeze())
                y_test_sub1 = torch.index_select( y_test_sub.squeeze(), dim=0, index = idx_test.squeeze())        
                
                h_loss_test = heteroscedastic_loss(y_test_sub1.squeeze(),y_test_pre1.squeeze(),y_test_pre_log_var1.squeeze())
                mse_loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())
                loss_test = mse_loss_test + heteroscedastic_loss_coefficient * h_loss_test
                MAE2 = mean_absolute_error(y_test_sub1.cpu().numpy().squeeze(),y_test_pre1.cpu().numpy().squeeze())
                r2_score_v =  r2_score(y_test_sub1.cpu().numpy().squeeze(),y_test_pre1.cpu().numpy().squeeze()) 
                
                if epoch % 20 == 0:
                    log_message = f'Iter-{epoch}; Total loss: {loss_test.item():.4f}; MAE2: {MAE2:.4f}; r2_score_v: {r2_score_v:.4f}'
                    log_and_print(log_message, log_file_path)

                if r2_best < r2_score_v:
                    best_test_loss = loss_test
                    torch.save(model.state_dict(), os.path.join(results_dir, 'best_test_model_ale_epi_.pth'))
                    MAE_best = MAE2
                    r2_best = r2_score_v
                
            loss_test_log.append(torch.mean(loss_test).item())
        
        model_test = UA_CNN().to(device)
        model_test.load_state_dict(torch.load(os.path.join(results_dir, 'best_test_model_ale_epi_.pth')))
        y_test_pre, y_test_vector, y_test_pre_log_var = model_test(x_test)
        y_train_pre, y_train_vector, y_train_pre_log_var= model_test(x_train)
        
        with torch.no_grad():
            idx_train = torch.nonzero(y_train_sub.squeeze()!=0,as_tuple=False)
            idx_tst = torch.nonzero(y_test_sub.squeeze()!=0,as_tuple=False)
            y_train_sub1 = torch.index_select(y_train_sub.squeeze(), dim=0, index = idx_train.squeeze())
            y_train_pre1 = torch.index_select(y_train_pre.squeeze(), dim=0, index = idx_train.squeeze())
            y_train_pre_log_var1 = torch.index_select(y_train_pre_log_var.squeeze(), dim=0, index = idx_train.squeeze())
            y_train_vector1 = torch.index_select(y_train_vector.squeeze(), dim=0, index = idx_train.squeeze())
            y_test_pre_log_var1 = torch.index_select(y_test_pre_log_var.squeeze(), dim=0, index = idx_tst.squeeze())
            y_test_sub1 = torch.index_select(y_test_sub.squeeze(), dim=0, index = idx_tst.squeeze())
            y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index = idx_tst.squeeze())
            y_test_vector1 = torch.index_select(y_test_vector.squeeze(), dim=0, index = idx_tst.squeeze())
            
        mse_loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())
        h_loss_test = heteroscedastic_loss(y_test_sub1.squeeze(),y_test_pre1.squeeze(),y_test_pre_log_var1.squeeze())
        loss_test = mse_loss_test + heteroscedastic_loss_coefficient * h_loss_test
        MAE_test = mean_absolute_error(y_test_sub1.cpu().numpy().squeeze(),y_test_pre1.cpu().numpy().squeeze())
        r2_test = r2_score(y_test_sub1.cpu().numpy().squeeze(),y_test_pre1.cpu().numpy().squeeze())
        
        mse_loss_train = loss_func(y_train_pre1.squeeze(), y_train_sub1.squeeze())
        h_loss_train = heteroscedastic_loss(y_train_sub1.squeeze(),y_train_pre1.squeeze(),y_train_pre_log_var1.squeeze())
        loss_train = mse_loss_train + heteroscedastic_loss_coefficient * h_loss_train
        MAE_train = mean_absolute_error(y_train_sub1.cpu().numpy().squeeze(),y_train_pre1.cpu().numpy().squeeze())
        r2_train = r2_score(y_train_sub1.cpu().numpy().squeeze(),y_train_pre1.cpu().numpy().squeeze())
        
        
        test_preds_array = np.array([[x] for x in y_test_pre1.cpu().numpy()])
        sum_preds += test_preds_array
        
        test_pred_log_vars_array = np.array([[x] for x in y_test_pre_log_var1.cpu().numpy()])
        test_pred_vars_array = np.exp(test_pred_log_vars_array)
        
        sum_ale_uncs += test_pred_vars_array
        
        all_preds[:, :, i_seed] = test_preds_array
            
        labels = y_test_pre1.cpu().numpy()
        
        if r2_test > r2_record:
            r2_record = r2_test
            torch.save(y_train_sub1.squeeze(), os.path.join(results_dir, 'y_train_sub_ale_epi_.pth'))
            torch.save(y_train_pre1.squeeze(), os.path.join(results_dir, 'y_train_pre_ale_epi_.pth'))
            torch.save(y_test_sub1.squeeze(), os.path.join(results_dir, 'y_test_sub_ale_epi_.pth'))
            torch.save(y_test_pre1.squeeze(), os.path.join(results_dir, 'y_test_pre_ale_epi_.pth'))  
            torch.save(y_test_vector.squeeze(), os.path.join(results_dir, 'y_test_vector_ale_epi_.pth'))
            torch.save(y_train_vector.squeeze(), os.path.join(results_dir, 'y_train_vector_ale_epi_.pth')) 
        
        result_r2_test.iloc[i_seed,i+1] = r2_test
        result_mae_test.iloc[i_seed,i+1] = MAE_test
        result_loss_test.iloc[i_seed,i+1] = loss_test.detach().cpu().numpy()
        result_r2_train.iloc[i_seed,i+1] = r2_train
        result_mae_train.iloc[i_seed,i+1] = MAE_train
        result_loss_train.iloc[i_seed,i+1] = loss_train.detach().cpu().numpy()
        
        i = i+1
        
        current_results = (f"Seed {seed} results: R2 test = {r2_test:.4f}, MAE test = {MAE_test:.4f}, "
                            f"Loss test = {loss_test:.4f}, R2 train = {r2_train:.4f}, "
                            f"MAE train = {MAE_train:.4f}, Loss train = {loss_train:.4f}")
        log_and_print(current_results, log_file_path)
    results.append(result_r2_test)
    results.append(result_mae_test)
    results.append(result_loss_test)
    results.append(result_r2_train)
    results.append(result_mae_train)
    results.append(result_loss_train)
    results_all.append(results)

    avg_preds = sum_preds / seeds_len
    avg_preds = avg_preds.tolist()

    avg_ale_uncs = sum_ale_uncs / seeds_len
    avg_ale_uncs = avg_ale_uncs.tolist()

    avg_epi_uncs = np.var(all_preds, axis=2)
    avg_epi_uncs = avg_epi_uncs.tolist()

    x= [i for i in range(len(labels))]
    y = labels
    f = avg_preds
    f = [item for sublist in f for item in sublist]
    ale_var = avg_ale_uncs
    ale_var = [item for sublist in ale_var for item in sublist]
    epi_var = avg_epi_uncs
    epi_var = [item for sublist in epi_var for item in sublist]
    x = np.array(x)
    y = np.array(y)
    f = np.array(f)
    epi_var = np.array(epi_var)
    ale_var = np.array(ale_var)
    total_std_2 = (epi_var + ale_var)**0.5
    epi_std = epi_var**0.5
    ale_std = ale_var**0.5
    total_std = epi_std + ale_std
    save_results_to_excel(results_excel_file, exp_num + 1, avg_preds, epi_std, ale_std, total_std, total_std_2, y)
    print(f"Finished saving results for experiment {exp_num + 1}")  
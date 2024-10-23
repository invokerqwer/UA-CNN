import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math
import time
import sys
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

results_list = []

csv_type = sys.argv[1]  # "physical_para" or "new_labels"
device = torch.device(f"cuda:{sys.argv[2]}" if torch.cuda.is_available() else "cpu")
XRD_descriptor_filename = sys.argv[3]
grid_points_type = sys.argv[4]

print(f"CSV Type: {csv_type}")
print(f"Device: {device}")
print(f"XRD Descriptor Filename: {XRD_descriptor_filename}")
print(f"Grid Points Type: {grid_points_type}")

from datetime import datetime
parent_dir = './x_grad_visiualize_exp/'
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
results_dir = parent_dir + 'results/x_grad/zhifangtu/new3/' + f'results_{csv_type}_{XRD_descriptor_filename}/'

import os
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

log_file = open(results_dir + 'log.txt', 'w')
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_file.write(f"Program start time: {start_time}\n")
log_file.write(f"Using device: {device}\n")
log_file.flush()

XRD_descriptor = pd.read_csv(parent_dir + XRD_descriptor_filename + '.csv', header=None, low_memory=False)

if csv_type == "physical_para":
    label_data = pd.read_csv('../data/934label-16_no-header.csv', header=None, low_memory=False)
elif csv_type == "new_labels":
    label_data = pd.read_csv('../data/data_no_header.csv', header=None, low_memory=False)
else:
    log_file.write(f"Invalid csv_type: {csv_type}\n")
    log_file.close()
    sys.exit(1)

if grid_points_type == "32_32":
    grid_points = [[32, 32]]
else:
    grid_points = [[12, 52]]

# 模型路径列表
model_saved_paths = [
    f'./results/best_test_model_saved_physical_para_col_{i}.pth'
    for i in range(16)
]
results_all = []
r2_record = -math.inf
best_seed = -1
seeds = [199228]

# 定义模型类
class UA_CNN(nn.Module):
    def __init__(self):
        super(UA_CNN, self).__init__()
        self.partition_1 = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        ])

        self.partition_2 = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        ])

        self.fc1 = nn.Linear(256, 64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(32, 8)
        self.relu7 = nn.ReLU()
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x, gp):
        x1 = x[:, :, :gp[0] * 64]
        x2 = x[:, :, gp[0] * 64:]

        for item in self.partition_1:
            x1 = item(x1)
        for item in self.partition_2:
            x2 = item(x2)

        x = torch.cat([x1, x2], dim=2)

        out = self.fc1(x)
        out = self.relu5(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        out = self.relu7(out)
        out = self.fc4(out)
        return out, x

attribute_names = [
    'PLD', 'LCD', 'AV', 'AV (cm^3/g)', 'POAV', 'POAV (cm^3/g)', 
    'ASA (m^2/cm^3)', 'ASA (m^2/g)', 'Density', 'CH4 (70 bar)', 
    'CO2 (3 bar)', 'CH4 (1 bar)', 'CO2 (1 bar)', 'a', 'b', 'c'
]

# Corresponding filenames (replace '/' with '_')
attribute_filenames = [
    'PLD', 'LCD', 'AV', 'AV_cm3_g', 'POAV', 'POAV_cm3_g', 
    'ASA_m2_cm3', 'ASA_m2_g', 'Density', 'CH4_70bar', 
    'CO2_3bar', 'CH4_1bar', 'CO2_1bar', 'a', 'b', 'c'
] 

def plot_gradients(gradient_means, model_index):
    plt.figure(figsize=(15, 10))  # 增加图形大小
    plt.bar(range(len(gradient_means)), gradient_means, alpha=0.75, color='blue', edgecolor='black')
    
    # 获取属性名称和文件名安全的版本
    attribute_name = attribute_names[model_index]
    attribute_filename = attribute_filenames[model_index]
  
    # 设置字体属性和标题，使用更大的字体
    plt.title(f'Gradient Mean Distribution for {attribute_name}', fontsize=38, fontname='DejaVu Sans')
    plt.xlabel('Feature Index', fontsize=38, fontname='DejaVu Sans')
    plt.ylabel('Gradient Mean Value', fontsize=38, fontname='DejaVu Sans')
    print(f'Gradient plot for {attribute_name},model_index:{model_index}')
   
    # 调整刻度字体大小
    plt.xticks(fontsize=28, fontname='DejaVu Sans')
    plt.yticks(fontsize=28, fontname='DejaVu Sans')
    
    plt.grid(True)
    
    # 保存高分辨率图像
    plt.savefig(results_dir + f'gradient_mean_distribution_{attribute_filename}.png', dpi=300)
    plt.close()

    # 将梯度均值保存到CSV文件
    gradient_data = pd.DataFrame({
        'Feature Index': list(range(len(gradient_means))),
        'Gradient Mean': gradient_means
    })
    csv_filename = results_dir + f'gradient_mean_distribution_{attribute_filename}.csv'
    gradient_data.to_csv(csv_filename, index=False)


# 函数：计算梯度统计
def compute_gradient_stats(gradients):
    gradients = gradients.cpu().detach().numpy()
    mean_per_feature = np.mean(gradients, axis=0).flatten()  # 每个特征的均值
    std_per_feature = np.std(gradients, axis=0).flatten()    # 每个特征的标准差
    return mean_per_feature, std_per_feature,gradients

# 对每个模型进行梯度计算和分析
for grid_point in tqdm(grid_points):
    results = []

    for i_seed in tqdm(range(len(seeds)), position=0, leave=True):
        seed = seeds[i_seed]
        setup_seed(seed)

        # 对于每个模型，设置不同的 column_index
        for model_index, model_path in enumerate(model_saved_paths):
            column_index = model_index  # 设置要预测的属性列
            data = pd.concat([XRD_descriptor, label_data.iloc[:, column_index]], axis=1)  # 更新y列

            # 使用整个数据集
            x_data = data.iloc[:, :4096]
            y_data = data.iloc[:, 4096]

            x_data = torch.from_numpy(x_data.values).float().to(device)
            y_data = torch.from_numpy(y_data.values).float().to(device)

            x_data = torch.unsqueeze(x_data, 1)

            model_test = UA_CNN().to(device)
            model_test.load_state_dict(torch.load(model_path))
            loss_func = nn.MSELoss()

            x_data.requires_grad = True
            model_test.train()  # 确保模型处于训练模式
            with torch.enable_grad():
                y_data_pre, y_data_vector = model_test(x_data, grid_point)

                print(f"Model Output: {y_data_pre[:5]}")  # 输出前几个预测结果进行检查

                # 计算损失
                idx_tst = torch.nonzero(y_data.squeeze() != 0, as_tuple=False)
                y_data_sub1 = torch.index_select(y_data.squeeze(), dim=0, index=idx_tst.squeeze())
                y_data_pre1 = torch.index_select(y_data_pre.squeeze(), dim=0, index=idx_tst.squeeze())
                
                print(f"y_data_sub1 Sample: {y_data_sub1[:5]}")  # 打印前几个真实值进行检查
                print(f"y_data_pre1 Sample: {y_data_pre1[:5]}")  # 打印前几个预测值进行检查

                loss_data = nn.MSELoss()(y_data_pre1.squeeze(), y_data_sub1.squeeze())
                  # 计算 MAE 和 R2
                MAE_test = mean_absolute_error(y_data_sub1.cpu().detach().numpy(), y_data_pre1.cpu().detach().numpy())
                r2_test = r2_score(y_data_sub1.cpu().detach().numpy(), y_data_pre1.cpu().detach().numpy())
              
                loss_data.backward()

                # 获取输入张量的梯度
                input_gradients = x_data.grad

                if torch.all(input_gradients == 0):
                    print(f"Warning: All gradients are zero for model {model_index}!")
                
                # 计算并记录梯度统计信息
                gradient_means, gradient_stds,gradients = compute_gradient_stats(input_gradients)
                # 绘制梯度分布
                # plot_gradients(gradient_means, gradient_stds, model_index)
                plot_gradients(gradient_means, model_index)

                print(f"Model: {model_path}")
                print(f"MAE Test: {MAE_test}")
                print(f"R2 Test: {r2_test}")
                print(f"Loss: {loss_data.item()}")  # 打印损失值检查
                print(f"Gradient Mean per Feature: {gradient_means[:10]}...")  # 仅打印前10个特征的均值示例
                print(f"Gradient Std per Feature: {gradient_stds[:10]}...")    # 仅打印前10个特征的标准差示例
                print('gradients.shape',gradients.shape)
                print('gradients',gradients)
                # 去掉多余的维度
                gradients_squeezed = np.squeeze(gradients)
                gradient_means_squeezed = np.squeeze(gradient_means)
                # 保存处理后的梯度
                np.savetxt(results_dir + f'gradients_model_{model_index}.csv', gradients_squeezed, delimiter=",")
                np.savetxt(results_dir + f'gradient_means_model_{model_index}.csv', gradients_squeezed, delimiter=",")


        results_all.append(results)

# 打印程序结束时间
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("Program end time:", end_time)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math
import time
import sys
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

results_list = []

csv_type = sys.argv[1]  # "physical_para" or "new_labels"
column_index = int(sys.argv[2])
device = torch.device(f"cuda:{sys.argv[3]}" if torch.cuda.is_available() else "cpu")
XRD_descriptor_filename = sys.argv[4]
grid_points_type = sys.argv[5]

print(f"CSV Type: {csv_type}")
print(f"Column Index: {column_index}")
print(f"Device: {device}")
print(f"XRD Descriptor Filename: {XRD_descriptor_filename}")
print(f"Grid Points Type: {grid_points_type}")

from datetime import datetime
parent_dir = './'
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
results_dir = parent_dir + 'results/' + f'results_{csv_type}_{column_index}_{XRD_descriptor_filename}/'

import os
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

log_file = open(results_dir + f'log_{csv_type}_col_{column_index}.txt', 'w')
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_file.write(f"Program start time: {start_time}\n")
log_file.write(f"Using device: {device}\n")
log_file.flush()

XRD_descriptor = pd.read_csv(parent_dir + XRD_descriptor_filename + '.csv', header=None, low_memory=False)

if csv_type == "physical_para":
    label_data = pd.read_csv('./data/934label-16_no-header.csv', header=None, low_memory=False)
elif csv_type == "new_labels":
    label_data = pd.read_csv('./data/data_no_header.csv', header=None, low_memory=False)
else:
    log_file.write(f"Invalid csv_type: {csv_type}\n")
    log_file.close()
    sys.exit(1)

data = pd.concat([XRD_descriptor, label_data.iloc[:, column_index]], axis=1)
if grid_points_type == "32_32":
    grid_points = [[32, 32]]
else:
    grid_points = [[12, 52]]

results_all = []
r2_record = -math.inf
best_seed = -1
seeds = [199228, 302675, 257057, 320858, 844620, 298933, 681403, 690678]

for grid_point in tqdm(grid_points):
    results = []

    for i_seed in tqdm(range(len(seeds)), position=0, leave=True):
        seed = seeds[i_seed]

        setup_seed(seed)

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=seed)
        
        x_train = train_data.iloc[:, :4096]
        x_test = test_data.iloc[:, :4096]
        y_train = train_data.iloc[:, 4096]
        y_test = test_data.iloc[:, 4096]

        x_train = torch.from_numpy(x_train.values).float().to(device)
        x_test = torch.from_numpy(x_test.values).float().to(device)
        y_train = torch.from_numpy(y_train.values).float().to(device)
        y_test = torch.from_numpy(y_test.values).float().to(device)

        x_train = torch.unsqueeze(x_train, 1)
        x_test = torch.unsqueeze(x_test, 1)

        input_size, feature_size = x_train.shape[0], x_train.shape[1]
        LR = 0.001
        epochs = 1000
        mb_size = 50
        loss_train_log = []
        loss_test_log = []

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

        def random_mini_batches(X_train, Y_train, mini_batch_size=10):
            mini_batches = []
            X_train = torch.split(X_train, mini_batch_size)
            Y_train = torch.split(Y_train, mini_batch_size)
            for i in np.arange(len(X_train)):
                mini_batch = (X_train[i], Y_train[i])
                mini_batches.append(mini_batch)
            return mini_batches

        model = UA_CNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.MSELoss()

        y_train_sub = y_train
        y_test_sub = y_test
        r2_best = -math.inf
        MAE_best = 0
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            num_minibatches = int(input_size / mb_size) + 1
            minibatches = random_mini_batches(x_train, y_train_sub, mb_size)
            model.train()
            for minibatch in minibatches:
                batch_x, batch_y = minibatch
                batch_y_pre, _ = model(batch_x, grid_point)
                idx = torch.nonzero(batch_y.squeeze() != 0, as_tuple=False)
                batch_y_pre1 = torch.index_select(batch_y_pre.squeeze(), dim=0, index=idx.squeeze())
                batch_y1 = torch.index_select(batch_y.squeeze(), dim=0, index=idx.squeeze())
                loss = loss_func(batch_y_pre1.squeeze(), batch_y1.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss = epoch_loss + (loss / num_minibatches)
            loss_train_log.append(torch.mean(epoch_loss).item())

            if epoch % 20 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    y_test_pre, _ = model(x_test, grid_point)
                    idx_test = torch.nonzero(y_test_sub.squeeze() != 0, as_tuple=False)
                    y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index=idx_test.squeeze())
                    y_test_sub1 = torch.index_select(y_test_sub.squeeze(), dim=0, index=idx_test.squeeze())
                    loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())

                    y_test_sub1_cpu = y_test_sub1.cpu().numpy()
                    y_test_pre1_cpu = y_test_pre1.cpu().numpy()

                    MAE2 = mean_absolute_error(y_test_sub1_cpu, y_test_pre1_cpu)
                    r2_score_v = r2_score(y_test_sub1_cpu, y_test_pre1_cpu)

                    if r2_best < r2_score_v:
                        best_test_loss = loss_test
                        torch.save(model.state_dict(), results_dir + f'best_test_model_{csv_type}_col_{column_index}.pth')
                        MAE_best = MAE2
                        r2_best = r2_score_v
                    epoch_end_time = time.time()
                    epoch_duration = epoch_end_time - epoch_start_time
                    log_file.write(f'Epoch {epoch}; Total loss: {loss_test.item():.4}; MAE2: {MAE2:.4}; r2_score_v: {r2_score_v:.4}; Duration: {epoch_duration:.2f} seconds\n')
                    log_file.flush()

            loss_test_log.append(torch.mean(loss_test).item())

        model_test = UA_CNN().to(device)
        model_test.load_state_dict(torch.load(results_dir + f'best_test_model_{csv_type}_col_{column_index}.pth'))
        y_test_pre, y_test_vector = model_test(x_test, grid_point)
        y_train_pre, y_train_vector = model_test(x_train, grid_point)

        with torch.no_grad():
            idx_train = torch.nonzero(y_train_sub.squeeze() != 0, as_tuple=False)
            idx_tst = torch.nonzero(y_test_sub.squeeze() != 0, as_tuple=False)
            y_train_sub1 = torch.index_select(y_train_sub.squeeze(), dim=0, index=idx_train.squeeze())
            y_train_pre1 = torch.index_select(y_train_pre.squeeze(), dim=0, index=idx_train.squeeze())
            y_test_sub1 = torch.index_select(y_test_sub.squeeze(), dim=0, index=idx_tst.squeeze())
            y_test_pre1 = torch.index_select(y_test_pre.squeeze(), dim=0, index=idx_tst.squeeze())
            y_test_vector1 = torch.index_select(y_test_vector.squeeze(), dim=0, index=idx_tst.squeeze())
            y_train_vector1 = torch.index_select(y_train_vector.squeeze(), dim=0, index=idx_train.squeeze())

        loss_test = loss_func(y_test_pre1.squeeze(), y_test_sub1.squeeze())
        MAE_test = mean_absolute_error(y_test_sub1.cpu().numpy(), y_test_pre1.cpu().numpy())
        r2_test = r2_score(y_test_sub1.cpu().numpy(), y_test_pre1.cpu().numpy())

        loss_train = loss_func(y_train_pre1.squeeze(), y_train_sub1.squeeze())
        MAE_train = mean_absolute_error(y_train_sub1.cpu().numpy(), y_train_pre1.cpu().numpy())
        r2_train = r2_score(y_train_sub1.cpu().numpy(), y_train_pre1.cpu().numpy())

        if r2_test > r2_record:
            r2_record = r2_test
            best_seed = seed
            torch.save(model.state_dict(), results_dir + f'best_test_model_saved_{csv_type}_col_{column_index}.pth')
            torch.save(y_train_sub1.squeeze(), results_dir + f'y_train_sub_{csv_type}_col_{column_index}.pth')
            torch.save(y_train_pre1.squeeze(), results_dir + f'y_train_pre_{csv_type}_col_{column_index}.pth')
            torch.save(y_test_sub1.squeeze(), results_dir + f'y_test_sub_{csv_type}_col_{column_index}.pth')
            torch.save(y_test_pre1.squeeze(), results_dir + f'y_test_pre_{csv_type}_col_{column_index}.pth')
            torch.save(y_test_vector.squeeze(), results_dir + f'y_test_vector_{csv_type}_col_{column_index}.pth')
            torch.save(y_train_vector.squeeze(), results_dir + f'y_train_vector_{csv_type}_col_{column_index}.pth')

        results_list.append({
            'Seed': seed,
            'Train_R2': r2_train,
            'Train_MAE': MAE_train,
            'Train_Loss': loss_train.item(),
            'Test_R2': r2_test,
            'Test_MAE': MAE_test,
            'Test_Loss': loss_test.item()
        })

        log_file.write(f"Results: R2 Test: {r2_test}, MAE Test: {MAE_test}, Loss Test: {loss_test}\n")
        log_file.write(f"Results: R2 Train: {r2_train}, MAE Train: {MAE_train}, Loss Train: {loss_train}\n")
        print(f"Results: Seed:{seed},R2 Test: {r2_test}, MAE Test: {MAE_test}, Loss Test: {loss_test}")
        log_file.flush()

    results_all.append(results)
print('Best r2_record:',r2_record, 'best_seed:',best_seed)
print(f"CSV Type: {csv_type}")
print(f"Column Index: {column_index}")
print(f"Device: {device}")
print(f"XRD Descriptor Filename: {XRD_descriptor_filename}")
print(f"Grid Points Type: {grid_points_type}")



results_df = pd.DataFrame(results_list)
result_path = results_dir + f'results_{csv_type}_{column_index}_{XRD_descriptor_filename}.csv'

mean_r2_test = results_df['Test_R2'].mean()
var_r2_test = results_df['Test_R2'].var()

mean_mae_test = results_df['Test_MAE'].mean()
var_mae_test = results_df['Test_MAE'].var()

mean_loss_test = results_df['Test_Loss'].mean()
var_loss_test = results_df['Test_Loss'].var()

print(f"Mean R2 Test: {mean_r2_test:.4f}, var R2 Test: {var_r2_test:.4f}")
print(f"Mean MAE Test: {mean_mae_test:.4f}, var MAE Test: {var_mae_test:.4f}")
print(f"Mean Loss Test: {mean_loss_test:.4f}, var Loss Test: {var_loss_test:.4f}")

results_df.loc['Summary'] = [
    column_index,mean_r2_test, var_r2_test, mean_mae_test, var_mae_test, mean_loss_test, var_loss_test
]

results_df.to_csv(result_path, index=False)
print(f"Updated result saved to: {result_path}")

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
log_file.write(f"Program end time: {end_time}\n")
log_file.write(f"Mean R2 Test: {mean_r2_test:.4f}, var R2 Test: {var_r2_test:.4f}\n")
log_file.write(f"Mean MAE Test: {mean_mae_test:.4f}, var MAE Test: {var_mae_test:.4f}\n")
log_file.write(f"Mean Loss Test: {mean_loss_test:.4f}, var Loss Test: {var_loss_test:.4f}\n")
log_file.close()
print("Program end time:", end_time)
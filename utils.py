import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
      
def extract_coords_to_csv(ax, csv_filename): 
    if ax is None:
        raise ValueError("The provided Axes object is None")
    calibration_line = ax.lines[1] 
    calibration_x = calibration_line.get_xdata()
    calibration_y = calibration_line.get_ydata()
    calibration_coords = list(zip(calibration_x, calibration_y))
    df = pd.DataFrame(calibration_coords, columns=["X", "Y"])
    df.to_csv(csv_filename, index=False)
    print(f"Calibration line coordinates saved to {csv_filename}")
    
def heteroscedastic_loss(true, mean, log_var):
    """
    Compute the heteroscedastic loss for regression.

    :param true: A list of true values.
    :param mean: A list of means (output predictions).
    :param log_var: A list of logvars (log of predicted variances).
    :return: Computed loss.
    """
    precision = torch.exp(-log_var)
    loss = precision * (true - mean)**2 + log_var
    return loss.mean()

def random_mini_batches(X_train, Y_train, mini_batch_size = 10):                           
    mini_batches = []
    X_train = torch.split(X_train, mini_batch_size)
    Y_train = torch.split(Y_train, mini_batch_size)
    for i in np.arange(len(X_train)):
        mini_batch = (X_train[i],Y_train[i])
        mini_batches.append(mini_batch)
    return mini_batches


def get_proportion_lists_vectorized(y_pred, y_std, y_true, prop_type="interval"):
    """
    Compute the expected and observed proportions of data points within prediction intervals.
    
    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        prop_type: "interval" to measure observed proportions for centered prediction intervals,
                   and "quantile" for observed proportions below a predicted quantile.
    
    Returns:
        exp_proportions: Expected proportions.
        obs_proportions: Observed proportions.
    """
    n_points = len(y_true)
    exp_proportions = np.linspace(0, 1, num=n_points)
    
    if prop_type == "interval":
        z_scores = np.linspace(-3, 3, num=n_points)
        pred_intervals = np.array([y_pred + z * y_std for z in z_scores])
        within_interval = (y_true[:, None] <= pred_intervals).mean(axis=0)
        obs_proportions = within_interval
    elif prop_type == "quantile":
        quantiles = np.linspace(0, 1, num=n_points)
        pred_quantiles = np.array([np.percentile(y_pred + q * y_std, 100 * q) for q in quantiles])
        below_quantile = (y_true[:, None] <= pred_quantiles).mean(axis=0)
        obs_proportions = below_quantile
    else:
        raise ValueError("Invalid prop_type. Use 'interval' or 'quantile'.")
    
    return exp_proportions, obs_proportions


import numpy as np
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct

import torch
def calculate_error_drop(y_true, y_pred, uncertainty, quantiles=10):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(uncertainty, np.ndarray):
        uncertainty = torch.tensor(uncertainty)
    errors = torch.abs(y_true - y_pred)
    
    sorted_indices = torch.argsort(uncertainty)
    sorted_errors = errors[sorted_indices]
    q = quantiles
    q_errors = torch.chunk(sorted_errors, q)
    
    error_drop = torch.mean(q_errors[0]) / torch.mean(q_errors[-1])
    
    return error_drop.item()


def calculate_decreasing_ratio(y_true, y_pred, uncertainty, quantiles=10):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(uncertainty, np.ndarray):
        uncertainty = torch.tensor(uncertainty)
    errors = torch.abs(y_true - y_pred)
    
    sorted_indices = torch.argsort(uncertainty)
    sorted_errors = errors[sorted_indices]
    
    q = quantiles
    q_errors = torch.chunk(sorted_errors, q)
    
    decreasing_count = 0
    for i in range(len(q_errors) - 1):
        if torch.mean(q_errors[i]) >= torch.mean(q_errors[i+1]):
            decreasing_count += 1
    
    decreasing_ratio = decreasing_count / (q - 1)
    
    return decreasing_ratio
def calculate_auco(y_true, y_pred, uncertainty, quantiles=10):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(uncertainty, np.ndarray):
        uncertainty = torch.tensor(uncertainty)

    errors = torch.abs(y_true - y_pred)
    sorted_indices = torch.argsort(uncertainty)
    sorted_errors = errors[sorted_indices]
    
    q = quantiles
    q_errors = torch.chunk(sorted_errors, q)
    
    oracle_sorted_indices = torch.argsort(errors)
    oracle_sorted_errors = errors[oracle_sorted_indices]
    oracle_q_errors = torch.chunk(oracle_sorted_errors, q)
    
    auco = 0
    for i in range(q - 1):
        auco += (torch.mean(q_errors[i]) - torch.mean(oracle_q_errors[i])).item()
    
    return auco

def evaluate_model_metrics(y_true, y_pred, uncertainty, quantiles=100):
    error_drop = calculate_error_drop(y_true, y_pred, uncertainty, quantiles)
    decreasing_ratio = calculate_decreasing_ratio(y_true, y_pred, uncertainty, quantiles)
    auco = calculate_auco(y_true, y_pred, uncertainty, quantiles)
    
    return {
        'Error Drop': error_drop,
        'Decreasing Ratio': decreasing_ratio,
        'AUCO': auco
    }


import numpy as np
import matplotlib.pyplot as plt

def calculate_cumulative_mae(y, f, uncertainty):
    sorted_indices = np.argsort(uncertainty)[::-1]
    sorted_uncertainty = uncertainty[sorted_indices]
    sorted_y = y[sorted_indices]
    sorted_f = f[sorted_indices]
    
    cumulative_mae = []
    n = len(y)
    for i in range(n):
        mae_i = np.mean(np.abs(sorted_f[i:] - sorted_y[i:]))
        cumulative_mae.append(mae_i)
    
    return cumulative_mae, sorted_uncertainty


def log_and_print(message, log_file_path):
    print(message)
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')
        
import numpy as np
import scipy.stats
from typing import List, Tuple
def cal_confidence_based_calibration_metrics(true_arr: np.ndarray,
                                             pred_arr: np.ndarray,
                                             unc_arr: np.ndarray,
                                             n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    data_size = len(true_arr)
    confidence_level = np.linspace(0, 1, n_bins + 1, endpoint=True)
    fractions = []
    
    for conf in confidence_level:
        count = 0
        for mean, var, true in zip(pred_arr, unc_arr, true_arr):
            lower_bound, upper_bound = scipy.stats.norm.interval(conf, loc=mean, scale=var**0.5)
            if lower_bound < true < upper_bound:
                count += 1
        fractions.append(count / data_size)

    return confidence_level, np.array(fractions)
def calculate_ece(confidence_levels: np.ndarray, empirical_fractions: np.ndarray) -> float:
    ece = np.mean(np.abs(confidence_levels - empirical_fractions))
    return ece
def calculate_miscalibration_area(confidence_levels: np.ndarray, empirical_fractions: np.ndarray) -> float:
    sorted_indices = np.argsort(confidence_levels)
    x = confidence_levels[sorted_indices]
    y = empirical_fractions[sorted_indices]

    perfect_y = x

    abs_diff = np.abs(y - perfect_y)

    miscalibration_area = np.trapz(abs_diff, x)

    return miscalibration_area


import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd

def plot_confidence_based_calibration_curve(y, f, uncertainty, uncertainty_type, output_dir, timestamp, n_bins=10):
    savefig=True
    save_metrics=True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    confidence_levels, fractions = cal_confidence_based_calibration_metrics(y, f, uncertainty, n_bins=n_bins)
    print("Confidence Levels:", confidence_levels)
    print("Fractions of true values in CI:", fractions)
    ece = calculate_ece(confidence_levels, fractions)
    print(f"ECE: {ece}")

    plt.figure(figsize=(6, 4))
    plt.plot(confidence_levels, fractions, label="Calibration Curve", color='r', linewidth=2)
    plt.plot([0, 1], [0, 1], '--', color='gray', label="Perfect Calibration")  
    plt.title(f"Confidence-based Calibration Curve, ECE: {ece:.5f}")
    plt.xlabel("Confidence Level")
    plt.ylabel("Fraction of true values in CI")
    plt.legend(loc="upper left")
    plt.grid(True)

    if savefig:
        plot_filename = os.path.join(output_dir, f"calibration_curve_{uncertainty_type}_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Calibration curve saved to {plot_filename}")

    plt.show()

    if save_metrics:
        data = {
            'Confidence Level': confidence_levels,
            'Fraction of true values in CI': fractions
        }
        df = pd.DataFrame(data)
        df['ECE'] = ece
        metrics_filename = os.path.join(output_dir, f"calibration_metrics_{uncertainty_type}_{timestamp}.csv")
        df.to_csv(metrics_filename, index=False)
        print(f"Calibration metrics saved to {metrics_filename}")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os

def plot_and_save_cumulative_mae(y, f, ale_std, epi_std, total_std, total_std_2, output_dir, timestamp):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cumulative_mae_ale, sorted_ale_std = calculate_cumulative_mae(y, f, ale_std)
    cumulative_mae_epi, sorted_epi = calculate_cumulative_mae(y, f, epi_std)
    cumulative_mae_total, sorted_total_std = calculate_cumulative_mae(y, f, total_std)
    cumulative_mae_total_2, sorted_total_std_2 = calculate_cumulative_mae(y, f, total_std_2)
    
    n = len(y)
    confidence_percentiles = np.arange(1, n + 1) / n
    plt.figure(figsize=(10, 6))
    plt.plot(confidence_percentiles, cumulative_mae_ale, marker='o', linestyle='-', label='ale_std')
    plt.plot(confidence_percentiles, cumulative_mae_epi, marker='s', linestyle='-', label='epi_std')
    plt.plot(confidence_percentiles, cumulative_mae_total, marker='^', linestyle='-', label='total_std')
    plt.plot(confidence_percentiles, cumulative_mae_total_2, marker='*', linestyle='-', label='total_std_2')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Cumulative MAE')
    plt.title('Cumulative MAE vs. Confidence Percentile')
    plt.grid(True)
    plt.legend()
    
    image_filename = os.path.join(output_dir, f"CumulativeMAE_vs_ConfidencePercentile_{timestamp}.png")
    plt.savefig(image_filename)
    print(f"Plot saved to {image_filename}")
    plt.show()
    
    def save_coords_to_csv(confidence_percentiles, cumulative_mae, uncertainty_type):
        df = pd.DataFrame({
            'Confidence Percentile': confidence_percentiles,
            'Cumulative MAE': cumulative_mae
        })
        csv_filename = os.path.join(output_dir, f"CumulativeMAE_vs_ConfidencePercentile_{uncertainty_type}_{timestamp}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Coordinates saved to {csv_filename}")
    
    save_coords_to_csv(confidence_percentiles, cumulative_mae_ale, 'ale_std')
    save_coords_to_csv(confidence_percentiles, cumulative_mae_epi, 'epi_std')
    save_coords_to_csv(confidence_percentiles, cumulative_mae_total, 'total_std')
    save_coords_to_csv(confidence_percentiles, cumulative_mae_total_2, 'total_std_2')


import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import datetime
import os

def calculate_and_save_correlations(y, f, epi_std, ale_std, total_std, total_std_2, output_dir, timestamp):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    abs_error = np.abs(y - f)
    
    spearman_epi, p_value_epi = spearmanr(epi_std, abs_error)
    spearman_ale, p_value_ale = spearmanr(ale_std, abs_error)
    spearman_total, p_value_total = spearmanr(total_std, abs_error)
    spearman_total_2, p_value_total_2 = spearmanr(total_std_2, abs_error)
    
    print("Spearman correlation epi coefficient:", spearman_epi)
    print("Spearman correlation ale coefficient:", spearman_ale)
    print("Spearman correlation total coefficient:", spearman_total)
    print("Spearman correlation total_2 coefficient:", spearman_total_2)
    
    pearson_epi, p_value_epi_p = pearsonr(epi_std, abs_error)
    pearson_ale, p_value_ale_p = pearsonr(ale_std, abs_error)
    pearson_total, p_value_total_p = pearsonr(total_std, abs_error)
    pearson_total_2, p_value_total_2_p = pearsonr(total_std_2, abs_error)
    
    print("Pearson correlation epi coefficient:", pearson_epi)
    print("Pearson correlation ale coefficient:", pearson_ale)
    print("Pearson correlation total coefficient:", pearson_total)
    print("Pearson correlation total_2 coefficient:", pearson_total_2)
    
    metrics_data = {
        'Spearman_epi': [spearman_epi],
        'Spearman_ale': [spearman_ale],
        'Spearman_total': [spearman_total],
        'Spearman_total_2': [spearman_total_2],
        'Pearson_epi': [pearson_epi],
        'Pearson_ale': [pearson_ale],
        'Pearson_total': [pearson_total],
        'Pearson_total_2': [pearson_total_2]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    csv_filename = os.path.join(output_dir, f"correlation_metrics_{timestamp}.csv")
    df_metrics.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")


import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save_calibration_metrics(pred_mean_list, pred_std_list, y, output_dir, timestamp):
    savefig = True
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics_list = []

    std_names = ["epi_std", "ale_std", "total_std", "total_std_2"][:len(pred_std_list)]

    for i, pred_mean in enumerate(pred_mean_list):
        for j, pred_std in enumerate(pred_std_list):
            std_name = std_names[j]

            # ----------------- Before Recalibration -----------------
            exp_props, obs_props = uct.get_proportion_lists_vectorized(pred_mean, pred_std, y)
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y, recal_model=None)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y, recal_model=None)
            ma = uct.miscalibration_area(pred_mean, pred_std, y, recal_model=None)
            print("Before Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

            metrics_list.append({
                'Type': std_name,
                'Recalibration': 'Before',
                'MACE': mace,
                'RMSCE': rmsce,
                'MA': ma
            })

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            uct.plot_calibration(pred_mean, pred_std, y, exp_props=exp_props, obs_props=obs_props, ax=ax)
            if savefig:
                csv_filename = os.path.join(output_dir, f"before_recal_{std_name}_{timestamp}.csv")
                extract_coords_to_csv(ax, csv_filename)
                uct.viz.save_figure(os.path.join(output_dir, f"before_recal_{std_name}_{timestamp}"), "svg")

            # ----------------- After Recalibration -----------------
            recal_model = uct.iso_recal(exp_props, obs_props)
            recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(
                pred_mean, pred_std, y, recal_model=recal_model)
            mace = uct.mean_absolute_calibration_error(pred_mean, pred_std, y, recal_model=recal_model)
            rmsce = uct.root_mean_squared_calibration_error(pred_mean, pred_std, y, recal_model=recal_model)
            ma = uct.miscalibration_area(pred_mean, pred_std, y, recal_model=recal_model)
            print("After Recalibration:  ", end="")
            print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

            metrics_list.append({
                'Type': std_name,
                'Recalibration': 'After',
                'MACE': mace,
                'RMSCE': rmsce,
                'MA': ma
            })

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            uct.plot_calibration(pred_mean, pred_std, y, exp_props=recal_exp_props, obs_props=recal_obs_props, ax=ax)
            if savefig:
                csv_filename = os.path.join(output_dir, f"after_recal_{std_name}_{timestamp}.csv")
                extract_coords_to_csv(ax, csv_filename)
                uct.viz.save_figure(os.path.join(output_dir, f"after_recal_{std_name}_{timestamp}"), "svg")

    df_metrics = pd.DataFrame(metrics_list)

    metrics_filename = os.path.join(output_dir, f"calibration_metrics_{timestamp}.csv")
    df_metrics.to_csv(metrics_filename, index=False)
    print(f"Metrics saved to {metrics_filename}")




import numpy as np
import matplotlib.pyplot as plt
import csv

def confidence_curve(y_true, y_pred, uncertainty, q=100):
    """
    Calculate the confidence curve.
    
    Parameters:
    y_true: Ground truth values
    y_pred: Predicted values
    uncertainty: Uncertainty estimates
    q: Number of quantiles (default 100)
    
    Returns:
    confidence_percentiles: Array of percentiles
    mean_absolute_errors: Corresponding MAE values at each percentile
    oracle_curve: The oracle (best possible confidence curve)
    """
    if len(y_true) != len(y_pred) or len(y_pred) != len(uncertainty):
        raise ValueError("Input arrays must have the same length")
    
    sorted_indices = np.argsort(-uncertainty)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    mae_full = np.abs(y_true_sorted - y_pred_sorted)
    mae_percentiles = []
    
    for i in range(1, q+1):
        n = int(len(mae_full) * (i / q))
        mae_percentiles.append(np.mean(mae_full[n:]))  
    
    sorted_by_error = np.argsort(-np.abs(y_true - y_pred))
    y_true_sorted_oracle = y_true[sorted_by_error]
    y_pred_sorted_oracle = y_pred[sorted_by_error]
    
    mae_full_oracle = np.abs(y_true_sorted_oracle - y_pred_sorted_oracle)
    oracle_percentiles = []
    
    for i in range(1, q+1):
        n = int(len(mae_full_oracle) * (i / q))
        oracle_percentiles.append(np.mean(mae_full_oracle[n:]))
    
    return np.arange(0, 100, 100/q), mae_percentiles, oracle_percentiles

def calculate_AUCO(mae_percentiles, oracle_percentiles):
    """
    Calculate the AUCO (Area Under Confidence Oracle).
    """
    arr = np.array(mae_percentiles) - np.array(oracle_percentiles)
    return np.sum(arr[:-1])

def calculate_error_drop(mae_percentiles):
    """
    Calculate the error drop (ratio between 1st and last quantile errors).
    """
    return mae_percentiles[0] / mae_percentiles[-2]

def calculate_decreasing_ratio(mae_percentiles):
    """
    Calculate the decreasing ratio, the fraction of non-increasing values in the curve.
    """
    non_increasing_count = sum(1 for j in range(len(mae_percentiles)-1) if mae_percentiles[j] >= mae_percentiles[j+1])
    return non_increasing_count / (len(mae_percentiles) - 1)

def plot_and_save_curve(percentiles, mae_percentiles, oracle_percentiles, filename='confidence_curve.png'):
    """
    Plot and save the confidence curve and oracle curve to a file.
    """
    plt.plot(percentiles, mae_percentiles, label="Confidence Curve")
    plt.plot(percentiles, oracle_percentiles, label="Oracle Curve", linestyle="--")
    plt.xlabel("Percentile")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def save_auco_metrics_to_csv(auco, error_drop, decreasing_ratio, filename='metrics.csv'):
    """
    Save AUCO, error drop, and decreasing ratio to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['AUCO', 'Error Drop', 'Decreasing Ratio'])
        writer.writerow([auco, error_drop, decreasing_ratio])

def process_confidence_curve(y_true, y_pred, uncertainty,output_dir, timestamp ):
    """
    Complete process to compute and save confidence curve and related metrics.
    """
    output_image= f'confidence_curve_{timestamp}.png'
    output_csv=f'auco_etrics_{timestamp}.csv'
    
    percentiles, mae_percentiles, oracle_percentiles = confidence_curve(y_true, y_pred, uncertainty)
    plot_and_save_curve(percentiles, mae_percentiles, oracle_percentiles, filename=os.path.join(output_dir,output_image))
    
    auco = calculate_AUCO(mae_percentiles, oracle_percentiles)
    error_drop = calculate_error_drop(mae_percentiles)
    decreasing_ratio = calculate_decreasing_ratio(mae_percentiles)
    
    print(f"AUCO: {auco}")
    print(f"Error Drop: {error_drop}")
    print(f"Decreasing Ratio: {decreasing_ratio}")
    save_auco_metrics_to_csv(auco, error_drop, decreasing_ratio, filename=os.path.join(output_csv))

import torch
import numpy as np
import os

def compute_stats(tensor):
    max_value = torch.max(tensor).detach().numpy()
    min_value = torch.min(tensor).detach().numpy()
    median_value = torch.median(tensor).detach().numpy()
    variance = torch.var(tensor).detach().numpy()
    return max_value, min_value, median_value, variance

def makedirs(path: str, isfile: bool = False):
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)
import rasterio 
import re
import os
import geopandas as gpd
from rasterio.transform import rowcol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats

def normalizingData(X, min_per, max_per):
    temp=(X - min_per) / (max_per - min_per)
    temp[temp>1]=1
    temp[temp<0]=0
    return temp

def plot_point_time_series(prediction_path, gt_path, save_path = None):
    """
    Plot and compare real and predicted soil moisture values.

    This function reads ground truth and predicted soil moisture values from CSV files from 
    a point in 2 year (2021-2022),creates a scatter and line plot for both, and either displays or saves the plot.
    - prediction_path: Path to the CSV file containing predicted values (column 'Prediction').
    - gt_path: Path to the CSV file containing ground truth values (column 'sm_25').
    - save_path: If provided, saves the plot to this path; otherwise, displays the plot.
    """
    predicted = pd.read_csv(prediction_path)
    gt = pd.read_csv(gt_path)

    # Extract real and predicted values
    real_values = gt['sm_25'] if 'sm_25' in gt.columns else None
    predicted_values = predicted['Prediction']

    df = {'real values' : real_values, 'predictions' : predicted_values}
    df = pd.DataFrame(df)

    # Create a point plot
    plt.figure(figsize=(12, 6))
    if real_values is not None:
        plt.scatter(range(len(real_values)), real_values, label='Real Values', color='blue', alpha=0.6)
        plt.plot(range(len(real_values)), real_values, color='blue', alpha=0.6, linestyle='--')
    plt.scatter(range(len(predicted_values)), predicted_values, label='Predicted Values', color='red', alpha=0.6)
    plt.plot(range(len(predicted_values)), predicted_values, color='red', alpha=0.6, linestyle='--')

    plt.xlabel('Pixel Index')
    plt.ylabel('Soil Moisture')
    plt.title('Real vs Predicted Soil Moisture Values')
    plt.legend()
    plt.ylim(0.1, 0.6)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_point_time_series(prediction_path, gt_path, smap_path, save_path = None):

    """
    Plot and compare real and predicted soil moisture values.

    This function reads ground truth, SMAP and predicted soil moisture values from CSV files from 
    a point in 2 year (2021-2022) in Thai Binh,creates a scatter and line plot for both, and either displays or saves the plot.
    - prediction_path: Path to the CSV file containing predicted values (column 'Prediction').
    - gt_path: Path to the CSV file containing ground truth values (column 'sm_25').
    - save_path: If provided, saves the plot to this path; otherwise, displays the plot.
    """
    # Load real and predicted values
    predicted = pd.read_csv(prediction_path)
    gt = pd.read_csv(gt_path)
    smap = pd.read_csv(smap_path)
    # Extract real and predicted values
    real_values = gt['sm_25'] if 'sm_25' in gt.columns else None #Soil moisture in 'sm_25' column
    date_list = gt['date'] if 'date' in gt.columns else None
    date_list = pd.to_datetime(date_list).to_list()
    predicted_values = predicted['Prediction']
    smap_values = smap['sm'] if 'sm' in smap.columns else None 
    smap_dates = smap['date'] if 'date' in smap.columns else None 
    smap_dates = pd.to_datetime(smap_dates).to_list()

    df = {'real values' : real_values, 'predictions' : predicted_values}
    df = pd.DataFrame(df)

    # Create a point plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(date_list, predicted_values, color='red', alpha=1.0, label = 'Predicted SM')
    if real_values is not None:
        ax.plot(date_list, real_values, color='blue', alpha=1.0, linestyle = '--', label = 'Planet SM')
    ax.plot(smap_dates, smap_values, color = 'green', alpha=1.0, linestyle = ':', label = 'SMAP')

    # Format x-axis as dates 
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation = 45)

    plt.xlabel('Pixel Index')
    plt.ylabel('Soil Moisture')
    plt.title('Real vs Predicted Soil Moisture Values')
    plt.legend()
    plt.ylim(0.1, 0.6)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_metrics(prediction_path, gt_path, save_path = None):
    """
    Calculate and plot regression metrics for real vs predicted soil moisture values.

    This function loads ground truth and predicted soil moisture values from CSV files,
    computes RMSE, unbiased RMSE (ubRMSE), MAE, and Pearson correlation, 
    fits and plots a trend line, and visualizes the results in a scatter plot.
    """
    # Load real and predicted values
    predicted = pd.read_csv(prediction_path)
    gt = pd.read_csv(gt_path)
    # Extract real and predicted values
    real_values = gt['sm_25'] if 'sm_25' in gt.columns else gt['sm'] #Soil moisture in 'sm_25' column
    predicted_values = predicted['Prediction']

    print(real_values.shape)

    mse = mean_squared_error(real_values, predicted_values)
    rmse = np.sqrt(mse)
    pearson_corr, pearson_p = scipy.stats.pearsonr(real_values, predicted_values)

    # Calculate unbiased RMSE (ubRMSE)
    bias = np.mean(predicted_values - real_values)
    ubrmse = np.sqrt(np.mean(((predicted_values - real_values) - bias) ** 2))

    # Calculate MAE
    mae = mean_absolute_error(real_values, predicted_values)

    # fit a trend line (linear regression) 
    z = np.polyfit(real_values, predicted_values, 1)
    p = np.poly1d(z)

    print(len(real_values))
    plt.figure(figsize = (6,6))
    plt.scatter(real_values, predicted_values, color='blue', alpha=0.6, s=1)
    plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], 'r--')
    plt.plot([0.1, 0.7], [0.1, 0.7], label='Ideal (y = x)', color = 'cyan')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Real vs Predicted Soil Moisture Values')
    plt.plot(real_values, p(real_values), "r-", label = 'Trend Line')

    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    print(f"RMSE: {rmse:.4f}")
    print(f"Unbiased RMSE (ubRMSE): {ubrmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson corr {pearson_corr:.4f}")

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

prediction_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/output/output_result.csv' 
gt_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/100m_data/output_tb/output_ndvi250_first/THAIBINH_101.csv'
smap_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/100m_data/output_tb/smap/130.csv'
plot_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/output/thaibinh_test_metrics.png'



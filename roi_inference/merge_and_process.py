import pandas as pd 
import datetime
import numpy as np
import rasterio
from rasterio.transform import xy 
import os 
from utils_data import get_coordinates, get_band_values, normalizingData
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
args = parser.parse_args()

region = args.region

root_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/roi_inference/regions_data_results'
os.makedirs(f'roi_inference/regions_data_results/{region}/prediction', exist_ok = True)
df_NDVI = pd.read_csv(f'{root_path}/{region}/csv_output/ndvi_values.csv')
df_T = pd.read_csv(f'{root_path}/{region}/csv_output/T_values.csv')
df_P = pd.read_csv(f'{root_path}/{region}/csv_output/P_values.csv')
# df_Aux = pd.read_csv(f'{root_path}/{region}/csv_output/soil_terrain_data.csv')
df_Aux = pd.read_csv(f'{root_path}/{region}/csv_output/extract_dem_soil_from_tif2.csv')

date_cols = df_NDVI.keys()[2:]
date_times_NDVI = pd.to_datetime(date_cols)

target_dates = []
s1_folder = f'roi_inference/regions_data_results/{region}/data/s1_images'

for s1_file in os.listdir(s1_folder):
    if s1_file.endswith('.tif'):
        date = s1_file.split('.')[0].split('_')[-1]
        target_dates.append(date)

first_image_path = f'{root_path}/{region}/data/s1_images/S1_{target_dates[0]}.tif'
coordinates = get_coordinates(first_image_path)
for target_date in target_dates:
    tif_path = f'{root_path}/{region}/data/s1_images/S1_{target_date}.tif'
    # coordinates = get_coordinates(tif_path)
    band_values = get_band_values(tif_path)  # shape: (3, num_pixels)

    print('Coordinates shape: ',coordinates.shape)  # Should print (height * width, 2)

    # Prepare DataFrame
    df_S1 = pd.DataFrame({
        'lon': coordinates[:, 0],
        'lat': coordinates[:, 1],
        'VV': band_values[0, :],
        'VH': band_values[1, :],
        'angle': band_values[2, :]
    })
    print('df_S1.head: ')
    print(df_S1.head())
    df_S1.to_csv(f'{root_path}/{region}/csv_output/s1/S1_{target_date}.csv', index=False)

    DoY = datetime.datetime.strptime(target_date, "%Y-%m-%d").timetuple().tm_yday 
    print('DoY: ', DoY)
    df_S1['DOY_sin'] = np.sin(DoY / 365 * np.pi * 2)
    df_S1['DOY_cos'] = np.cos(DoY / 365 * np.pi * 2)

    s1_date = pd.to_datetime(target_date)

    # From Sentinel-1 date, we can calculate the date range for NDVI and weather data. 
    # From the NDVI date nearest to the Sentinel-1 date, we select the last 364 days of NDVI data, that ensures we can get 46 values of NDVI.
    mask = date_times_NDVI <= s1_date 
    last_date_NDVI = date_times_NDVI[mask][-1]
    start_date_NDVI = last_date_NDVI - pd.Timedelta(days = 363)

    # Select the columns within the desired date range 
    selected_dates = date_times_NDVI[(date_times_NDVI >= start_date_NDVI) & (date_times_NDVI <= last_date_NDVI)]

    # Convert back to string to match DataFrame column names
    selected_date_cols = selected_dates.strftime('%Y-%m-%d').tolist()

    # Keep only the selected columns in df_NDVI 
    # df_NDVI.columns[:2] for the first two columns 'lon' and 'lat'
    cols_to_keep = list(df_NDVI.columns[:2]) + selected_date_cols

    df_NDVI_filtered = df_NDVI[cols_to_keep]
    df_T_filtered = df_T[cols_to_keep]
    df_P_filtered = df_P[cols_to_keep]

    # Normalize NDVI, T, and P values (excluding 'lon' and 'lat' columns)
    ndvi_values = df_NDVI_filtered.iloc[:, 2:].values
    t_values = df_T_filtered.iloc[:, 2:].values
    p_values = df_P_filtered.iloc[:, 2:].values


    # Copy the DataFrames to avoid modifying the original ones
    df_NDVI_filtered = df_NDVI_filtered.copy()
    df_T_filtered = df_T_filtered.copy()
    df_P_filtered = df_P_filtered.copy()

    """ Normalize the data so that it is in the range of 0 to 1 for each feature. With: 
    MODIS NDVI: -2000 to 10000
    ERA5 Temperature: 273.15 to 318.15 (Kelvin) in Vietnam
    ERA5 Precipitation: 0 to 0.3 (mm) in Vietnam
    """
    df_NDVI_filtered.loc[:, df_NDVI_filtered.columns[2:]] = normalizingData(ndvi_values, -2000, 10000)
    df_T_filtered.loc[:, df_T_filtered.columns[2:]] = normalizingData(t_values, 273.15, 318.15)
    df_P_filtered.loc[:, df_P_filtered.columns[2:]] = normalizingData(p_values, 0, 0.3)

    merged = df_S1.copy()
    # Exclude 'lon' and 'lat' from other DataFrames before merging, we don't need them for running inference
    dfs_to_merge = [
        df_Aux.drop(columns=['lon', 'lat']).replace(0.0, np.nan).replace(-32768.0, np.nan),
        df_NDVI_filtered.drop(columns=['lon', 'lat']),
        df_T_filtered.drop(columns=['lon', 'lat']),
        df_P_filtered.drop(columns=['lon', 'lat'])
    ]

    for df in dfs_to_merge:
        merged = pd.concat([merged, df], axis=1)

    n_cols = merged.shape[1]

    # Rename columns from the third onwards to '0', '1', '2', ..., 'n-3'
    new_cols = list(merged.columns[:2]) + [str(i) for i in range(n_cols - 2)]
    merged.columns = new_cols 

    # Min values and max values for normalization of other features from: Sentinel-1, SoilGrids, DEM 
    min_per = np.asarray([-30, -35, 29.1, -1, -1, 0, 0, 100, 0, 0, -1, -1, -1, -1, -1])
    max_per = np.asarray([5, 0, 46, 1, 1, 1000, 1000, 180, 5500, 40, 1, 1, 1, 1, 1])

    # Normalize the columns from index 2 to 16 (VV, VH, angle, soil texture, DEM, etc.) except 'lon' and 'lat' the first two columns
    cols_to_normalize = merged.columns[2:17]
    merged[cols_to_normalize] = merged[cols_to_normalize].astype(float)

    normalized = normalizingData(
        merged.loc[:, cols_to_normalize].values,
        min_per,
        max_per
    ).astype(float)

    merged.loc[:, cols_to_normalize] = normalized

    # merged.dropna(subset = ['5'], inplace = True)
    # merged = merged.ffill() #uncomment this line if you want to fill NaN values with the previous value in the column,not recommended 
    merged.to_csv(f'{root_path}/{region}/csv_output/combination/{target_date}_tif.csv', index=False)
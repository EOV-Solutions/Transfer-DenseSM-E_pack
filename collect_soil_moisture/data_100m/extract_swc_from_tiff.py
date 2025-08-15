import rasterio 
import numpy as np
import pandas as pd
import re
import os
import csv
import geopandas as gpd
from rasterio.transform import rowcol  
from datetime import datetime, timedelta

# region = 'china'  # or 'india'
def extract_data_for_region(region, station_df_path, root_path, tiff_folder, s1_date_csv, site_info_path, output_folder):
    """Get pixels' values and information of points as save them as CSV files 
    In this code, a point is called a station"""
    
    network = region.upper()
    s1_date = pd.read_csv(s1_date_csv)
    station_df = pd.read_csv(station_df_path)
    if not {'id', 'latitude', 'longitude'}.issubset(station_df.columns):
        raise ValueError("CSV file must contain 'id', 'latitude', and 'longitude' columns.")
    
    results = []
    
    for image in os.listdir(tiff_folder):
        if not image.endswith('.tiff'):
            continue
            
        date = re.search(r'(\d{4}-\d{2}-\d{2})', image)
        date = date.group(1) if date else None

        if date is None:
            continue  
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        previous_date = (date_obj - timedelta(days=1)).strftime('%Y-%m-%d')

        # Check if the previous date's image exists in the folder 
        if previous_date in s1_date['date'].values:
            date = previous_date
        else:
            print(f'There is data for S1 for this date: {date}')

        tiff_path = os.path.join(tiff_folder, image)
        with rasterio.open(tiff_path) as src:
            transform = src.transform
            
            for _, row in station_df.iterrows():
                latitude = row['latitude']
                longitude = row['longitude']
                station_id = row['id']

                # Convert latitude and longitude to row and column indices to get pixel values = 0.5 * sm
                # The first band is the quality flag (0 = good)
                # The second band is the soil moisture value. 
                # No need to check the third band.
                try:
                    row_index, col_index = rowcol(transform, longitude, latitude)
                    quality_flag = src.read(1)[row_index, col_index]
                    if quality_flag != 0: # 0 = good quality
                        pixel_value = np.nan
                        continue
                    pixel_value = src.read(2)[row_index, col_index] 
                    sm_25 = round(pixel_value * 2, 5) # sm = 2*pixel_value (band 2)
                except IndexError:
                    pixel_value = np.nan 
                
                results.append({
                    'id': int(station_id),
                    'date': date,
                    'latitude': latitude,
                    'longitude': longitude,
                    'pixel_value': pixel_value,
                    'sm': sm_25
                }) 

    # Convert results to DataFrame 
    results_df = pd.DataFrame(results)
    # Save results to CSV
    output_csv_path = f'{root_path}/{region}/{region}_swc_values.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved soil water content values to {output_csv_path}")
    
    create_site_info(results_df, network, site_info_path)
    
    # Create individual CSV files for each station
    create_data_csv_files(results_df, output_folder)

# Create site info file for all points: network, station (id), latitude, longitude, start depth, end depth)
# site info file will be used for downloading data in data_pre/Prepare_samples.ipynb
def create_site_info(df, network, output_csv):
    # Select unique locations 
    site_df = df[['id', 'latitude', 'longitude']].drop_duplicates().sort_values(by='id')

    # Add 'Network' and 'station' columns 
    site_df['network'] = network
    site_df['station'] = site_df['id']
    site_df['s_depth'] = 0 
    site_df['e_depth'] = 5
    
    # Reorder and rename columns for consistency
    site_df = site_df[['network', 'station', 'latitude', 'longitude', 's_depth', 'e_depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})
    
    site_df.to_csv(output_csv, index=False)
    print(f'Site info saved to {output_csv}')

# Create individual CSV files for each station with soil moisture data
# These file will be used for downloading data in data_pre/Prepare_samples.ipynb
def create_data_csv_files(df, output_folder):
    df['time'] = pd.to_datetime(df['date'])
    df['DoY'] = df['time'].dt.dayofyear
    df['station'] = df['id']

    for station, station_df in df.groupby('station'):
        station_file = os.path.join(output_folder, f'{int(station)}.csv')
        station_df['sm_count'] = 1
        station_df.to_csv(station_file, index=False)
        print(f"Saved: {station_file}")

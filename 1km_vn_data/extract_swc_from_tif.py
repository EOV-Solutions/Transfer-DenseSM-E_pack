
import rasterio 
import numpy as np
import pandas as pd
import re
import os
import csv
from pyproj import Transformer 
import geopandas as gpd
from rasterio.transform import rowcol  
from datetime import datetime, timedelta

tiff_folder = '/mnt/data2tb/nsidc_images'
root_path = '/mnt/data2tb/Transfer-DenseSM-E_2/1km_data'

def process_region(region='crop_wood'):
    shapefile_path = f'{root_path}/vn_points/{region}_points/{region}_points.shp'
    gdf = gpd.read_file(shapefile_path)

    gdf['latitude'] = gdf.geometry.y
    gdf['longitude'] = gdf.geometry.x

    csv_file_path = f'{root_path}/vn_points/{region}_points/{region}_points.csv'
    df = gdf.drop(columns='geometry').rename(columns={'field_1': 'id'})
    df.to_csv(csv_file_path, index=False)

    station_df = pd.read_csv(csv_file_path)

    if not {'id', 'latitude', 'longitude'}.issubset(station_df.columns):
        raise ValueError("CSV file must contain 'id', 'latitude', and 'longitude' columns.")

    coords = list(zip(station_df['longitude'], station_df['latitude']))
    results = []


    for image in os.listdir(tiff_folder):
        if not image.endswith('.tif'):
            continue

        match = re.search(r'(\d{8})', image)
        date = match.group(1) if match else None
        if date is None:
            continue
        date = datetime.strptime(date, '%Y%m%d')

        tiff_path = os.path.join(tiff_folder, image)
        with rasterio.open(tiff_path) as src:
            print(f"Processing image: {image}")
            print("Raster CRS:", src.crs)
            raster_crs = src.crs

            # Transform (lon, lat) to raster CRS
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            coords_lonlat = list(zip(station_df['longitude'], station_df['latitude']))
            coords_raster = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]

            # Sample values at transformed coordinates
            values = list(src.sample(coords_raster))

            for i, val in enumerate(values):
                if val is None or len(val) < 2 or np.ma.is_masked(val[0]) or np.ma.is_masked(val[1]):
                    sm_25 = np.nan
                else:
                    sm_25 = 0.5*(val[0] + val[1])

                row = station_df.iloc[i]
                results.append({
                    'id': int(row['id']),
                    'date': date,
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'sm': sm_25
                })

        print(f"Processed: {image}")

    # Convert results to DataFrame 
    results_df = pd.DataFrame(results)
    # Save results to CSV
    output_csv_path = f'{root_path}/vn_points/{region}_swc_values_2020.csv'
    results_df = results_df.dropna(subset = ['sm'])
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved soil water content values to {output_csv_path}")
    
    # Get information of each site (value, location)
    network = 'VN'
    print('results columns:', results_df.columns)
    create_site_info(results_df, region, network)
    
    # Create individual CSV files for each station
    create_data_csv_files(results_df, region)
    

def create_site_info(df, region, network):
    # Select unique locations 
    site_df = df[['id', 'latitude', 'longitude']].drop_duplicates().sort_values(by='id')

    # Add 'Network' and 'station' columns 
    site_df['network'] = network
    site_df['station'] = site_df['id']
    site_df['s_depth'] = 0 
    site_df['e_depth'] = 5

    site_df = site_df[['network', 'station', 'latitude', 'longitude', 's_depth', 'e_depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})
    
    output_csv = f'{root_path}/vn_points/{region}_site_info_2020.csv'
    site_df.to_csv(output_csv, index=False)
    print(f'Site info saved to {output_csv}')

def create_data_csv_files(df, region):
    df['time'] = pd.to_datetime(df['date'])
    df['DoY'] = df['time'].dt.dayofyear
    df['station'] = df['id']
    
    output_folder = f'{root_path}/vn_points/{region}_cvs'
    os.makedirs(output_folder, exist_ok=True)

    for station, station_df in df.groupby('station'):
        station_file = os.path.join(output_folder, f'{int(station)}.csv')
        station_df['sm_count'] = 1
        station_df = station_df.dropna(subset = ['sm'])
        if len(station_df) == 0:
            continue
        station_df = station_df.sort_values(by = 'date')
        station_df.to_csv(station_file, index=False)
        print(f"Saved: {station_file}")

if __name__ == "__main__":
    process_region()


import rasterio 
import numpy as np
import pandas as pd
import re
import os
import csv
from pyproj import Transformer 
import geopandas as gpd
from rasterio.transform import rowcol  
from datetime import datetime

def extract_and_create_files(points_csv_path, tiff_folder, site_info_path, sm_csv_folder, network):
    """ Here, region is the type of land cover, which we want to extract soil moisture data for.
        The function will read the shapefile of points (where we will get soil moisture), 
        extract soil moisture data from tiff files based on the points,
        and save the results to a CSV file.
    """
    station_df = pd.read_csv(points_csv_path)

    if not {'id', 'latitude', 'longitude'}.issubset(station_df.columns):
        raise ValueError("CSV file must contain 'id', 'latitude', and 'longitude' columns.")

    coords = list(zip(station_df['longitude'], station_df['latitude']))
    results = []

    # Loop through each image in the tiff folder (NSIDC source), get soil moisture values at the points
    for image in os.listdir(tiff_folder):
        if not image.endswith('.tif'):
            continue
        
        # Extract date from the image filename
        match = re.search(r'(\d{8})', image)
        date = match.group(1) if match else None
        if date is None:
            continue
        date = datetime.strptime(date, '%Y%m%d')

        # Open the tiff image and extract soil moisture values
        tiff_path = os.path.join(tiff_folder, image)
        with rasterio.open(tiff_path) as src:
            print(f"Processing image: {image}")
            print("Raster CRS:", src.crs)
            raster_crs = src.crs

            # Because tiff file is in EPSG:6933, we need to transform the coordinates of points to the raster CRS
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            coords_lonlat = list(zip(station_df['longitude'], station_df['latitude']))
            # Transform points' coordinates to raster CRS
            coords_raster = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]

            # Get values (soil moisture) at transformed coordinates
            # This will return a list of values for each point, each point has two values (am and pm soil moisture) 
            values = list(src.sample(coords_raster))

            # Loop through each point and get the average soil moisture values = 0.5 * (am + pm)
            for i, val in enumerate(values):
                # There are not enough 2 soil moisture values (am and pm) in a point, set to NaN
                if val is None or len(val) < 2 or np.ma.is_masked(val[0]) or np.ma.is_masked(val[1]):
                    sm_25 = np.nan
                # If there are 2 values, calculate the average for a day
                else:
                    sm_25 = 0.5*(val[0] + val[1])

                row = station_df.iloc[i]

                # Save soil moisture value and other information
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
    
    # Get information of each site (value, location)
    print('results columns:', results_df.columns)
    create_site_info(results_df, network, site_info_path)
    
    # Create individual CSV files for each station
    # These files will be used for data retrieving in file data_pre/Prepare_samples.ipynb
    create_data_csv_files(results_df, sm_csv_folder)
    

def create_site_info(df, network, site_info_path):
    """ Create a CSV file containing site information (id, latitude, longitude) for each point in the region.\
        The site information will be used for data retrieving in file data_pre/Prepare_samples.ipynb."""
    
    # Drop duplicate points  
    site_df = df[['id', 'latitude', 'longitude']].drop_duplicates().sort_values(by='id')

    # Add 'Network' and 'station' columns 
    site_df['network'] = network
    site_df['station'] = site_df['id']
    site_df['s_depth'] = 0 
    site_df['e_depth'] = 5

    site_df = site_df[['network', 'station', 'latitude', 'longitude', 's_depth', 'e_depth']].rename(
        columns={'latitude': 'lat', 'longitude': 'lon'})
    
    # Sort sites based on their station
    site_df = site_df.sort_values(by = 'station')
    site_df.to_csv(site_info_path, index=False)
    print(f'Site info saved to {site_info_path}')

def create_data_csv_files(df, output_folder):
    """ for each point (station) we create a CSV file containing:
      + soil moisture
      + coordinates
      + date
      ....
    """
    df['time'] = pd.to_datetime(df['date'])
    df['DoY'] = df['time'].dt.dayofyear
    df['station'] = df['id']
    
    os.makedirs(output_folder, exist_ok=True)

    # For each station (point)), create a CSV file containing soil moisture data
    for station, station_df in df.groupby('station'):
        station_file = os.path.join(output_folder, f'{int(station)}.csv')
        station_df['sm_count'] = 1
        station_df = station_df.dropna(subset = ['sm'])
        if len(station_df) == 0:
            continue
        station_df = station_df.sort_values(by = 'date')
        station_df.to_csv(station_file, index=False)
        print(f"Saved: {station_file}")
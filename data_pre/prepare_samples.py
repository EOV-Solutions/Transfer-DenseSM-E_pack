"""
Prepare samples using GEE
For each site, extract the full time series of Sentinel-1,NDVI data from the GEE. Note: if the output csv files already exist they are assumed to be correct and are not over-written.
"""

import os
import pandas as pd
import numpy as np
import time
import ee
import utils_data_pre_s3 as utils_data_pre
import yaml

def merge_filtered_sm_csv(sm_csv_folder, output_path, network):
    """
    Merge all filtered soil moisture CSV files in the specified folder into a single CSV file.
    The merged file will contain unique rows based on the 'sm' column.
    
    Input:
    - sm_csv_folder: Folder containing individual CSV files of soil moisture data.
    - output_path: Path to save the merged CSV file.
    - network: Name of the dataset.
    """
    # Load all CSV files of all sites the region (network)
    files = os.listdir(sm_csv_folder)

    # Initialize list to store DataFrames
    df_list = []

    # Traverse through each file in the directory
    for file in files:
        station = file.split('.')[0]

        # Read CSV
        df = pd.read_csv(os.path.join(sm_csv_folder, file))
        # Check if the first column is unnamed or empty, and drop it if necessary
        if df.columns[0] in [None, '', 'Unnamed: 0']:
            df = df.iloc[:, 1:] # drop the first column

        # Drop rows with NaN values
        df = df.dropna()

        # Insert 'network' and 'station' columns at the beginning
        df.insert(0, 'network', network)
        df.insert(1, 'station', station)
        # print(len(df))
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index= True)

    merged_df.insert(0, 's_index', range(1, len(merged_df)+1))

    # Save to a single csv file
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged soil moisture data in {output_path} with {len(merged_df)} samples")


"""
A loop to prepare the input data of each site

Step: 1 Create the grid polygon covering a site in both EASE2.0 and WGS84
Step: 2 Extract Sentienl-1, soil texture, terrain, NDVI, precipition, temperature etc. Check the utils for the details
Step: 3 Concatenate all data
Step: 4 Extract the surface soil moisture of the site

Note: 
* 'CHINA_100m': 100m dataset from Planet for the China's region.
* 'CHINA_1km' : 1km dataset from NSIDC for the China's region.
* 'INDIA_100m': 100m dataset from Planet for the India's region. 
* 'INDIA_1km' : 1km dataset from NSIDC for the India's region.
* 'VN'        : 1km dataset from NSIDC for Vietnam.
"""
def prepare_samples_for_sites(sm_sites, dir_to_site_sm, dir_to_site_samples, grid_size, network):

    # Create the grid to get input data based on sites(points)' coordinates
    pobj=utils_data_pre.grids_4_a_region(4326, grid_size) 
    # Read the sites information and determine the grid size
    # Read the site information, includ the lat, lon, and the site name
    sites = pd.read_csv(sm_sites, float_precision="high")
    
    filtered_sites = sites[sites['network'] == network] # filter the sites by network
    filtered_sites.reset_index(drop = True, inplace=True)
    print("Some some sites' information:")
    print(filtered_sites.head())

    # Loop through each site to prepare the samples
    for i in range(len(filtered_sites)):
        site = sites.loc[i]
        print(f"Processing for {i}/{len(filtered_sites)}: {site['station']}")
        # Create the path to save the samples
        path_2_site_file = os.path.join(dir_to_site_samples,'%s.csv'%(site['network']+'_'+str(site['station'])))

        # Check if the file already exists, if so, skip to the next site
        if os.path.exists(path_2_site_file):
            print(f"{path_2_site_file} is already done.")
            continue

        # Create the polygon grid covering the site in both EASE2.0 and WGS84
        ring_wgs,grid_ring=pobj.get_wgs_grid(site.lon,site.lat)
        polygon_grid=ee.Geometry.Polygon(ring_wgs, 'EPSG:4326', True, 20, False)

        # Extract the samples for the site
        samples,df_S1=utils_data_pre.samples_4_grid_v1(polygon_grid,START_DATE, END_DATE,START_DATE_NDVI,END_DATE_NDVI,ring_wgs,pobj, grid_size)
        if df_S1 is None or samples is None:
            print("Abort")
            continue

        # include the ground truth of soil moisture
        station_sm=pd.read_csv(os.path.join(dir_to_site_sm,'%s.csv'%(str(site['station']))),parse_dates=['time'])
        sm_point=station_sm[station_sm.time.dt.date.isin(list(df_S1.date.dt.date))]['sm']

        # Check if the length of sm_point matches the length of df_S1, we will get dates that are in df_S1
        if len(sm_point) != len(df_S1):
            print(f'Value and key do not have the same length, it is not a problem! {len(sm_point)} vs {len(df_S1)}')

        sm_df = pd.DataFrame({'date': station_sm.time.dt.date, 'sm': sm_point})
        sm_df['date'] = pd.to_datetime(sm_df['date'])
        # Merge df_S1 with the soil moisture data, we will keep the dates that are in df_S1
        df_S1 = df_S1.merge(sm_df, on = 'date', how = 'left')
        # df_S1.loc[df_S1.date.dt.date.isin(list(station_sm.time.dt.date)),'sm_25']=list(sm_point)

        # Concatenate the samples (NDVI, Temperature, Precipation. SoilGrids, DEM data) with df_S1 (Sentinel-1 data)
        try:
            samples=pd.DataFrame(samples,index=df_S1.index)
        except Exception as e:
            print(f"Error creating DataFrame: {e}")
            print(f"samples shape: {np.shape(samples)}")
            print(f"df_S1 shape: {df_S1.shape}")
            continue
        samples=pd.concat([df_S1,samples],axis=1)
        samples.dropna(inplace = True)
        samples.to_csv(path_2_site_file)
        # Sleep for a while to avoid GEE errors
        time.sleep(5)
        print("Done !!!")


# Network & Region
NETWORK = "VN"   # có thể lấy từ argparse để chạy CLI
with open("data_pre/regions.yaml", "r") as f:
    REGIONS = yaml.safe_load(f)

cfg = REGIONS[NETWORK]

# Home data directory
HOME_DATA_DIR = cfg["home"]

# Sentinel-1 date range
START_DATE = str(cfg["start_date"])
END_DATE = str(cfg["end_date"])

# NDVI & weather date range
START_DATE_NDVI = str(cfg["start_date_ndvi"])
END_DATE_NDVI = str(cfg["end_date_ndvi"])

# Directories
SM_SITES = os.path.join(HOME_DATA_DIR, cfg["sites_file"])
DIR_TO_SITE_SM = os.path.join(HOME_DATA_DIR, cfg["sm_folder"])
DIR_TO_SITE_SAMPLES = os.path.join(HOME_DATA_DIR, cfg["samples_folder"])
MERGED_CSV_PATH = os.path.join(HOME_DATA_DIR, cfg["merged_file"])
os.makedirs(DIR_TO_SITE_SAMPLES, exist_ok=True)

# if resolution = 1km, then the grid size = 1.0, if resolution = 100, then grid size = 0.1
GRID_SIZE = 1.0  # km

if __name__ == "__main__":
    prepare_samples_for_sites(SM_SITES, DIR_TO_SITE_SM, DIR_TO_SITE_SAMPLES, GRID_SIZE, NETWORK)
    merge_filtered_sm_csv(DIR_TO_SITE_SAMPLES, MERGED_CSV_PATH, NETWORK)
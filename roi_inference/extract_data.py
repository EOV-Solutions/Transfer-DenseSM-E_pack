import rasterio
from rasterio.transform import xy 
import numpy as np 
import os 
import pandas as pd
import utils_data
from utils_data import extract_terrain_soil_texture, get_coordinates, extract_ndvi, extract_era5
from rasterio.mask import mask
import argparse
from shapely.geometry import Polygon, Point, box
import pyproj
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()

# data_folder = '/mnt/data2tb/Transfer-DenseSM-E_pack/roi_inference/regions_data_results'
parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
parser.add_argument('--data_folder', required = True)
args = parser.parse_args()

region = args.region
data_folder = args.data_folder

# Create necessary directories to store output CSV files extracted from the image data
os.makedirs(f'{data_folder}/{region}/csv_output', exist_ok=True)
os.makedirs(f'{data_folder}/{region}/csv_output/s1', exist_ok=True)
os.makedirs(f'{data_folder}/{region}/csv_output/combination', exist_ok=True)
s1_folder = f"{data_folder}/{region}/data/s1_images"

# Get the path to a Sentinel-1 image -> this is used to extract coordinates of all pixels that we want to extract data for
for s1_file in os.listdir(s1_folder):
    if s1_file.endswith('.tif'):
        date = s1_file.split('.')[0].split('_')[-1]
        tif_path = f'{data_folder}/{region}/data/s1_images/S1_{date}.tif'
        break

coordinates = get_coordinates(tif_path)
print("Got all S1 pixels'coordinates!!!")

"""Extract NDVI data"""
samples = []
ndvi_folder = f'{data_folder}/{region}/data/ndvi_images'
era5_folder = f'{data_folder}/{region}/data/weather_images'

df_ndvi, dates, num_ndvi_images = extract_ndvi(ndvi_folder, coordinates)
# Save to CSV
df_ndvi.to_csv(f"{data_folder}/{region}/csv_output/ndvi_values.csv", index=False)
print("Saved NDVI csv!")

"""Extract Temperature and Precipitation data from ERA5-Land"""
df_T, df_P = extract_era5(era5_folder, coordinates, dates, num_ndvi_images)

# Save temperature to CSV
df_T.to_csv(f"{data_folder}/{region}/csv_output/T_values.csv", index=False)
print("Saved Temperature csv!")

# Save precipitation to CSV
df_P.to_csv(f"{data_folder}/{region}/csv_output/P_values.csv", index=False)
print("Saved Precipitation csv!")

"""Extract soil texture and DEM data for all dates"""
# Path to the soilgrids and dem images
grid_size = 0.1
point_geometry = utils_data.PointGeometry(4326, 6933)
pobj = utils_data.grids_4_a_region(4326, grid_size)

df_static = extract_terrain_soil_texture(
    soil_tif = f"{data_folder}/{region}/data/soilgrid_images/SoilGrid_sand_clay_bdod.tif",
    dem_tif = f"{data_folder}/{region}/data/dem_images/DEM_elevation_slope_aspect_10m.tif",
    coordinates = coordinates,
    pobj=pobj,
    grid_size=grid_size, 
    pg = point_geometry)

df_static.to_csv(f"{data_folder}/{region}/csv_output/extract_dem_soil_from_tif2.csv", index=False)
import os 
import numpy as np
import ee 
import pandas as pd 
import rasterio 
from rasterio.transform import Affine
from shapely.geometry import shape
import requests  
import geopandas as gpd
import argparse
from rasterio.enums import Resampling
import json
import re

ee.Initialize()

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', required=True)
parser.add_argument('--end_date', required=True)
parser.add_argument('--roi_geometry', required=False)
parser.add_argument('--roi_path', required=False)  # Thêm roi_path, không bắt buộc
parser.add_argument('--save_folder', required = True)
parser.add_argument('--region', required=True)

args = parser.parse_args()
save_folder = args.save_folder

def get_s1_dates_from_folder(s1_folder):
    """
    Lấy danh sách ngày từ tên các file S1 trong thư mục.
    Args:
        s1_folder (str): Đường dẫn tới thư mục chứa file S1.
    Returns:
        list: Danh sách ngày ở định dạng 'YYYY-MM-DD'.
    """
    date_strings = []
    for fname in os.listdir(s1_folder):
        if fname.startswith('S1_') and fname.endswith('.tif'):
            # Giả sử tên file dạng S1_YYYY-MM-DD.tif
            try:
                date_str = fname.split('_')[1].split('.')[0]
                date_strings.append(date_str)
            except Exception:
                continue
    date_strings = sorted(date_strings)
    return date_strings

def get_coordinates_from_shapefile(shapefile_path):
    """
    Extract coordinates from the first geometry in a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        list: List of (lon, lat) tuples representing the coordinates.
    """
    gdf = gpd.read_file(shapefile_path)
    geometry = gdf.geometry.iloc[0]
    if geometry.geom_type == 'Polygon':
        coords = list(geometry.exterior.coords)
    elif geometry.geom_type == 'MultiPolygon':
        coords = list(geometry.geoms[0].exterior.coords)
    else:
        raise ValueError("Geometry type not supported.")
    return coords

def extract_dates_from_filename(filename):
    match = re.search(r'(\d{8})', filename)
    if match:
        date_raw = match.group(1)
        return f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
    return None

# Get Sentinel-1 data for the specified geometry and date range
def get_S1(geometry, START_DATE, END_DATE, S1_DIR=''):
    datetime_range = f"{START_DATE}T00:00:00Z/{END_DATE}T23:59:59Z"

    payload = {
        "collection": ["sentinel-1-grd"],
        "geometry": geometry,
        "dateRange": datetime_range,
        "bands" : ["vv", "vh", "angle"]
    }

    """
    Thêm
    Code for downloading Sentinel-1 data, stacking bands and save in S1_DIR
    """
    # Rename tif files of S1 like format  S1_YYYY-MM-DD.tif
    for fname in os.listdir(S1_DIR):
        if fname.endswith('.tif'):
            date_str = extract_dates_from_filename(fname)
            if date_str:
                new_name = f"S1_{date_str}.tif"
                os.rename(os.path.join(S1_DIR, fname), os.path.join(S1_DIR, new_name))
    

# Download NDVI data from MOD13Q1 and MYD13Q1 collections. 16 day composite => 8 days interval
def download_NDVI_13Q1(geometry,START_DATE_NDVI, END_DATE_NDVI, NDVI_dir):
    datetime_range = f"{START_DATE_NDVI}T00:00:00Z/{END_DATE_NDVI}T23:59:59Z"

    payload = {
        "collection": ["MODIS/061/MOD13Q1", "MODIS/061/MYD13Q1"],
        "geometry": geometry,
        "dateRange": datetime_range,
        "bands": ["NDVI"]
    }

    """
    Thêm
    Code for downloading NDVI data and save in NDVI_dir
    """
    # Rename tif files of S1 like format  NDVI_YYYY-MM-DD.tif
    for fname in os.listdir(NDVI_dir):
        if fname.endswith('.tif'):
            date_str = extract_dates_from_filename(fname)
            if date_str:
                new_name = f"NDVI_{date_str}.tif"
                os.rename(os.path.join(NDVI_dir, fname), os.path.join(NDVI_dir, new_name))

# Download weather data from ERA5-Land for the specified polygon and date range
def download_weather_data(polygon_grid, start_date, end_date, weather_dir):
    """
    Download weather data from ERA5-Land for the specified polygon and date range
    """
    datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    payload = {
        "collection": ["ECMWF/ERA5_LAND/DAILY_AGGR"],
        "geometry": polygon_grid,
        "dateRange": datetime_range
    }

    """
    Thêm
    Code for downloading weather data, stacking bands 
    """
    # Rename tif files of weather data like format Weather_YYYY-MM-DD.tif
    for fname in os.listdir(weather_dir):
        if fname.endswith('.tif'):
            date_str = extract_dates_from_filename(fname)
            if date_str:
                new_name = f"Weather_{date_str}.tif"
                os.rename(os.path.join(weather_dir, fname), os.path.join(weather_dir, new_name))

# Download soil grid (sand, clay, bdod) and DEM images for the given geometry as GeoTIFFs
def download_SoilGrid_DEM_images(polygon_grid, soilgrid_dir, dem_dir):
    """
    Download soil grid (sand, clay, bdod) and DEM  images for the given geometry as GeoTIFFs
    """
    # SoilGrids payload 
    payload_soil = {
        "collection": ["SoilGrids"],
        "geometry": polygon_grid,
        "bands" : ["sand_0-5cm_mean", "clay_0-5cm_mean", "bdod_0-5cm_mean"]
    }

    payload_dem = {
        "collection": ["NASA/DEM"],
        "geometry": polygon_grid,
        "bands": ["elevation", "slope", "aspect"]
    }

    """
    Thêm
    Code for downloading soil grid and DEM images, stacking  bands and save in their respective directories
    """
    # Rename the soil grid file, chỉ đổi tên file đầu tiên tìm thấy và dừng lại
    for fname in os.listdir(soilgrid_dir):
        if fname.endswith('.tif'):
            src = os.path.join(soilgrid_dir, fname)
            dst = os.path.join(soilgrid_dir, "SoilGrid_sand_clay_bdob.tif")
            if src != dst:
                os.rename(src, dst)
            break  # Chỉ đổi tên một file duy nhất

    # Rename the DEM file, there is only one file
    for fname in os.listdir(dem_dir):
        if fname.endswith('.tif'):
            src = os.path.join(dem_dir, fname)
            dst = os.path.join(dem_dir, "DEM_elevation_slope_aspect_10m.tif")
            if src != dst:
                os.rename(src, dst)
            break  # Chỉ đổi tên một file duy nhất

def resample_s1_tifs_to_100m(s1_folder, output_folder=None):
    """
    Resample all S1 GeoTIFF images in a directory to 100m resolution.
    Args:
        s1_folder (str): Directory containing S1 .tif files.
        output_folder (str, optional): Directory to save resampled files. If None, overwrite originals.
    """
    for fname in os.listdir(s1_folder):
        if fname.startswith('S1_') and fname.endswith('.tif'):
            input_path = os.path.join(s1_folder, fname)
            output_path = os.path.join(output_folder or s1_folder, fname)
            with rasterio.open(input_path) as src:
                scale_factor = 100 / src.res[0]  # src.res[0] is pixel size (assume square)
                new_width = int(src.width * src.res[0] / 100)
                new_height = int(src.height * src.res[1] / 100)
                data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                new_transform = src.transform * src.transform.scale(
                    (src.width / new_width),
                    (src.height / new_height)
                )
                profile = src.profile.copy()
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': new_transform
                })
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data)

def get_data_from_s3(polygon_grid, start_date, end_date, region):
    """
    Fetch data from Google Earth Engine for a given polygon and date range.
    
    Args:
        polygon_grid (ee.Geometry): The polygon defining the area of interest.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        ee.ImageCollection: The image collection for the specified region and date range.
    """
    S1_dir = f'{save_folder}/{region}/data/s1_images_10m'
    DEM_dir = f'{save_folder}/{region}/data/dem_images'
    SoilGrid_dir = f'{save_folder}/{region}/data/soilgrid_images'
    NDVI_dir = f'{save_folder}/{region}/data/ndvi_images'
    weather_dir = f'{save_folder}/{region}/data/weather_images'
    landcover_dir = f'{save_folder}/{region}/data/land_cover'

    os.makedirs(S1_dir, exist_ok=True)
    os.makedirs(DEM_dir, exist_ok=True)
    os.makedirs(SoilGrid_dir, exist_ok=True)
    os.makedirs(NDVI_dir, exist_ok=True)
    os.makedirs(weather_dir, exist_ok=True)
    os.makedirs(landcover_dir, exist_ok = True)
    
    # Get Sentinel-1 data, returns a list of date strings
    get_S1(polygon_grid, start_date, end_date, S1_dir)
    date_strings = get_s1_dates_from_folder(S1_dir)
    
    # Get the first and last dates from the list of date strings to use for NDVI and weather data download
    first_s1_date = date_strings[0]
    last_s1_date = date_strings[-1]

    # Calculate the date range for NDVI data, one year before the first S1 date and four days after the last S1 date (ensure we have enough NDVI data for S1 dates)
    last_ndvi_date = (pd.to_datetime(last_s1_date) + pd.Timedelta(days=4)).strftime('%Y-%m-%d')
    first_ndvi_date = (pd.to_datetime(first_s1_date) - pd.Timedelta(days=368)).strftime('%Y-%m-%d')

    # Call functions to download NDVI, weather, soil grid, DEM, and land cover data
    download_NDVI_13Q1(polygon_grid, first_ndvi_date, last_ndvi_date, NDVI_dir)
    download_weather_data(polygon_grid, first_ndvi_date, last_ndvi_date, weather_dir)
    download_SoilGrid_DEM_images(polygon_grid, SoilGrid_dir, DEM_dir)

    # Resample Sentinel-1 images to 100 m resolution 
    resampled_S1_DIR = f'{save_folder}/{region}/data/s1_images'
    os.makedirs(resampled_S1_DIR, exist_ok=True)
    resample_s1_tifs_to_100m(S1_dir, resampled_S1_DIR)


if isinstance(args.roi_geometry, str):
    roi_geom = json.loads(args.roi_geometry)
else:
    roi_geom = args.roi_geometry

# Lấy bbox [xmin, ymin, xmax, ymax]
shp = shape(roi_geom)
bbox = list(shp.bounds)  # [xmin, ymin, xmax, ymax]
print("Bounding box:", bbox)

get_data_from_s3(polygon_grid = bbox, 
                 start_date=args.start_date, 
                 end_date=args.end_date, 
                 region=args.region)
print("Data fetched successfully.")


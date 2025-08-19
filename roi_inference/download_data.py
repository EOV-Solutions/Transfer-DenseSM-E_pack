import os 
import numpy as np
import ee 
import pandas as pd 
import rasterio 
from rasterio.transform import Affine
import requests  
import geopandas as gpd
import argparse

ee.Initialize()

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', required=True)
parser.add_argument('--end_date', required=True)
parser.add_argument('--roi_path', required=True)
parser.add_argument('--save_folder', required = True)
parser.add_argument('--region', required=True)

args = parser.parse_args()
save_folder = args.save_folder

def get_coordinates_from_tif(tif_path):
    """
    Extract coordinates from a GeoTIFF file.
    
    Args:
        tif_path (str): Path to the GeoTIFF file.
        
    Returns:
        tuple: A tuple containing the coordinates (lon, lat).
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds

        corners = [
            [bounds.left, bounds.top], # Top-left corner,
            [bounds.right, bounds.top], # Top-right corner,
            [bounds.right, bounds.bottom], # Bottom-right corner,
            [bounds.left, bounds.bottom], # Bottom-left corner
        ]
    print(corners)
    return corners

def export_image_to_drive(image, region, description, folder, scale, crs = 'EPSG:4326'):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Export task started: {description} to folder {folder}")

def download_tif(image, geometry, folder, image_id, resolution, crs = 'EPSG:4326'):
    local_path = os.path.join(folder, f"{image_id}.tif")
    # This check is to avoid re-downloading if the file of the date already exists
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping")
        return
    url = image.getDownloadURL({
        'scale': resolution,
        'crs': crs,
        'region': geometry,
        'filePerBand': False, 
        'format': 'GeoTIFF',
    })

    local_path = os.path.join(folder, f"{image_id}.tif")
    print(f"Downloading {image_id} to {local_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad requests 
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {image_id} successfully.")
    except Exception as e:
        print(f"Failed to download {image_id}: {e}")

# Get Sentinel-1 data for the specified geometry and date range
def get_S1(geometry, START_DATE, END_DATE, S1_DIR=''):
    ORBIT_PASS_A = 'ASCENDING'
    ORBIT_PASS_D = 'DESCENDING'

    # Get ASCENDING pass
    im_collection_a = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(geometry) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS_A)) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH', 'angle']) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 

    # Get DESCENDING pass
    im_collection_d = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(geometry) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.eq('orbitProperties_pass', ORBIT_PASS_D)) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH', 'angle']) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 

    # Merge both collections of ascending and descending passes
    im_collection = im_collection_a.merge(im_collection_d)

    # Sort by date
    im_collection = im_collection.sort('system:time_start')

    image_list = im_collection.toList(im_collection.size())

    size = im_collection.size().getInfo()

    print(f"Found {size} images between {args.start_date} and {args.end_date}")

    # Download each image as a GeoTIFF
    for i in range(im_collection.size().getInfo()):
        image = ee.Image(image_list.get(i))
        date_str = image.date().format('YYYY-MM-dd').getInfo()
        image_id = f"S1_{date_str}"
        download_tif(image, geometry, S1_DIR, image_id, 100)

    # Get list of acquisition dates so that we can use it later for NDVI and weather data download
    dates = im_collection.sort('system:time_start').aggregate_array('system:time_start')
    # Convert timestamps to formatted date strings
    date_strings = ee.List(dates).map(lambda d: ee.Date(d).format('YYYY-MM-dd')).getInfo()

    return date_strings

# Download NDVI data from MOD13Q1 and MYD13Q1 collections. 16 day composite => 8 days interval
def download_NDVI_13Q1(geometry,START_DATE_NDVI, END_DATE_NDVI, NDVI_dir):
    MOD_PRODUCT = "MODIS/061/MOD13Q1"
    bands = ['NDVI']
    collections_1 = ee.ImageCollection(MOD_PRODUCT) \
        .filterBounds(polygon_grid) \
        .filterDate(START_DATE_NDVI, END_DATE_NDVI) \
        .select(bands)

    MYD_PRODUCT = "MODIS/061/MYD13Q1"
    collection_2 = ee.ImageCollection(MYD_PRODUCT) \
        .filterBounds(polygon_grid) \
        .filterDate(START_DATE_NDVI, END_DATE_NDVI) \
        .select(bands)
    
    # Merge the two collections
    ndvi_collection = collections_1.merge(collection_2)
    image_list = ndvi_collection.toList(ndvi_collection.size())
    size = ndvi_collection.size().getInfo()
    # Download each image as a GeoTIFF
    for i in range(size):
        image = ee.Image(image_list.get(i))
        date_str = image.date().format('YYYY-MM-dd').getInfo()
        image_id = f"NDVI_{date_str}"
        download_tif(image, geometry, NDVI_dir, image_id, 100)

# Download weather data from ERA5-Land for the specified polygon and date range
def download_weather_data(polygon_grid, start_date, end_date, weather_dir):
    print(f"Extracting weather data from {start_date} to {end_date} for the specified polygon.")
    # Implement the logic to fetch and save weather data here
    ERA5_PRODUCT = 'ECMWF/ERA5_LAND/DAILY_AGGR'
    bands = ['temperature_2m', 'total_precipitation_sum']

    weather_collection = ee.ImageCollection(ERA5_PRODUCT) \
        .filterBounds(polygon_grid) \
        .filterDate(start_date, end_date) \
        .select(bands)
    
    image_list = weather_collection.toList(weather_collection.size())
    size = weather_collection.size().getInfo()
    # Download each image as a GeoTIFF
    for i in range(size):
        image = ee.Image(image_list.get(i))
        date_str = image.date().format('YYYY-MM-dd').getInfo()
        image_id = f"Weather_{date_str}"
        download_tif(image, polygon_grid, weather_dir, image_id, 1000)

# Download soil grid (sand, clay, bdod) and DEM images for the given geometry as GeoTIFFs
def download_SoilGrid_DEM_images(polygon_grid, soilgrid_dir, dem_dir):
    """
    Download soil grid (sand, clay, bdod) and DEM  images for the given geometry as GeoTIFFs
    """
    os.makedirs(soilgrid_dir, exist_ok = True)
    os.makedirs(dem_dir, exist_ok = True)

    # SoilGrid data
    sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')  # .select(sand);
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')  # .select(clay);
    bdod = ee.Image("projects/soilgrids-isric/bdod_mean").select('bdod_0-5cm_mean')  # .select(bdod);
    soil_image = sand.addBands([clay, bdod])
    soil_image_id = "SoilGrid_sand_clay_bdod"

    # DEM and terrain
    srtm = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Algorithms.Terrain(srtm).select(['elevation', 'slope', 'aspect']).toFloat()
    dem_image_id = "DEM_elevation_slope_aspect_10m"

    download_tif(soil_image, polygon_grid, soilgrid_dir, soil_image_id, 250)
    try: 
        download_tif(terrain, polygon_grid, dem_dir, dem_image_id, 30)
    # if the download fails because the file is too large, export to Google Drive instead
    except Exception as e:
        export_image_to_drive(terrain, polygon_grid, dem_image_id, 'GEE_Exports', 30)

# Download land cover image for visualization not for training or inference
def download_landcover_image(polygon_grid, landcover_dir):
    START = '2022-01-01'
    END = '2022-12-31'
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterDate(START, END) \
            .filterBounds(polygon_grid) \
            .select('label') \
            .mode()
    image_id = 'land_cover_100m'
    download_tif(dw, polygon_grid, landcover_dir, image_id, 100)

def get_data_from_ee(polygon_grid, start_date, end_date, region):
    """
    Fetch data from Google Earth Engine for a given polygon and date range.
    
    Args:
        polygon_grid (ee.Geometry): The polygon defining the area of interest.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        ee.ImageCollection: The image collection for the specified region and date range.
    """
    S1_dir = f'{save_folder}/{region}/data/s1_images'
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
    date_strings = get_S1(polygon_grid, start_date, end_date, S1_dir)
    print(date_strings)
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
    # Just for visualization, not for training or inference
    download_landcover_image(polygon_grid, landcover_dir)


# Determine the polygon of the region of interest, with the maximum allowed error is 20.
ring_wgs = get_coordinates_from_tif(args.roi_path)
polygon_grid=ee.Geometry.Polygon(ring_wgs, None, False)
get_data_from_ee(polygon_grid, args.start_date, args.end_date, args.region)
print("Data fetched successfully.")
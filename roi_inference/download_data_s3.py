import os 
import numpy as np
import ee 
import pandas as pd 
import rasterio 
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import shape
import geopandas as gpd
import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', required=True)
parser.add_argument('--end_date', required=True)
parser.add_argument('--roi_geometry', required=False)
parser.add_argument('--roi_path', required=False)  # Thêm roi_path, không bắt buộc
parser.add_argument('--save_folder', required = True)
parser.add_argument('--region', required=True)

args = parser.parse_args()
save_folder = args.save_folder

SERVER_IP = "10.0.0.60"   # thay IP server]
SERVER_PORT = 8000

COLLECTION_CONFIG = {
    "S1": ["vv", "vh", "angle"],
    "Weather": ["temperature_2m", "total_precipitation_sum"],
    "SoilGrid": ["sand_0-5cm_mean", "clay_0-5cm_mean", "bdod_0-5cm_mean"],
    "DEM": ["elevation", "slope", "aspect"]
}

def download_task(task_id, server_ip, server_port=8000, save_dir="downloads"):
    """
    Tải dữ liệu task_id từ FastAPI server về client và giải nén.
    """
    # URL download
    download_url = f"http://{server_ip}:{server_port}/v1/s1_download/{task_id}"

    # Tạo thư mục lưu
    os.makedirs(save_dir, exist_ok=True)
    zip_path = os.path.join(save_dir, f"{task_id}.zip")

    # Bước 1: tải file zip
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"[OK] Đã tải: {zip_path}")

def group_bands(folder, collection):
    suffixes = COLLECTION_CONFIG[collection]
    groups = defaultdict(dict)
    for fname in os.listdir(folder):
        if fname.lower().endswith(".tif"):
            for s in suffixes:
                pattern = re.compile(rf"_{s}\.tif$", re.IGNORECASE)
                if pattern.search(fname):
                    # bỏ luôn phần _vv.tif / _VV.tif, chỉ giữ lại tên gốc không có đuôi
                    base = pattern.sub("", fname)
                    groups[base][s] = os.path.join(folder, fname)
    return groups

# def group_bands(folder, collection):
#     suffixes = COLLECTION_CONFIG[collection]
#     groups = defaultdict(dict)
#     for fname in os.listdir(folder):
#         if fname.lower().endswith(".tif"):
#             for s in suffixes:
#                 # Regex: tìm _vv.tif hoặc _VV.tif (không phân biệt hoa/thường)
#                 pattern = re.compile(rf"_{s}\.tif$", re.IGNORECASE)
#                 if pattern.search(fname):
#                     base = pattern.sub(".tif", fname)  # thay _vv.tif / _VV.tif thành .tif
#                     groups[base][s] = os.path.join(folder, fname)
#     return groups

def stack_group(base, band_paths, out_dir, collection):
    """Stack theo band order trong COLLECTION_CONFIG"""

    band_order = COLLECTION_CONFIG[collection]
    arrays = []
    for b in band_order:
        with rasterio.open(band_paths[b]) as src:
            arrays.append(src.read(1))
            profile = src.profile

    stacked = np.stack(arrays, axis=0)  # (bands, h, w)
    out_name = f"{base}.tif"
    out_path = os.path.join(out_dir, out_name)

    profile.update(count=len(arrays))
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(len(arrays)):
            dst.write(stacked[i], i + 1)
    return out_path

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

# def unzip_and_cleanup(folder, task_id, remove_json=True):
#     """
#     Giải nén file zip theo task_id vào thư mục con, sau đó xóa zip gốc và (option) json.
    
#     Args:
#         folder (str): Thư mục chứa file zip.
#         task_id (str): Tên task id (không kèm .zip).
#         remove_json (bool): Có xóa file .json hay không.
    
#     Returns:
#         str: Đường dẫn tới thư mục đã giải nén.
#     """
#     zip_path = os.path.join(folder, f"{task_id}.zip")
#     extract_path = os.path.join(folder, task_id)
#     os.makedirs(extract_path, exist_ok=True)

#     # Extract zip
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)

#     # Xóa zip gốc
#     if os.path.exists(zip_path):
#         os.remove(zip_path)

#     # Xóa file .json (nếu có)
#     if remove_json:
#         for fname in os.listdir(extract_path):
#             if fname.endswith(".json"):
#                 os.remove(os.path.join(extract_path, fname))

#     return extract_path

def unzip_and_cleanup(folder, task_id, remove_json=True):
    zip_path = os.path.join(folder, f"{task_id}.zip")
    extract_path = os.path.join(folder, task_id)
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Bỏ prefix thư mục gốc (task_id/)
            filename = member.split("/", 1)[-1] if "/" in member else member
            if filename:  # tránh entry rỗng
                target = os.path.join(extract_path, filename)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zip_ref.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    if os.path.exists(zip_path):
        os.remove(zip_path)

    if remove_json:
        for fname in os.listdir(extract_path):
            if fname.endswith(".json"):
                os.remove(os.path.join(extract_path, fname))

    return extract_path

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
    Code for downloading Sentinel-1 data
    Sau khi chạy truy vấn sẽ thu được một task id từ server
    Code sẽ download một file zip đặt tên theo task id, 
    Sau khi unzip, folder này chứa tất cả các ảnh tif cần download được chia ra theo từng band, 
    (cùng một file zip và metadata jsson, nhưng chỉ cần quan tâm đến các ảnh)
    """
    task_id = "3a585547-6355-41a2-a24e-d3633194f356" # task id mô phỏng
    download_task(task_id, SERVER_IP, server_port=8000, save_dir=S1_DIR)
    # Tải về xong sẽ thu được thư mục tif và các file ảnh tif được extract ra từ các file zip => xóa file zip
    zip_path = os.path.join(S1_DIR, f"{task_id}.zip")
    extract_path = unzip_and_cleanup(S1_DIR, task_id)

    # Stack các band thảnh 1 ảnh đa bands duy nhất
    groups = group_bands(extract_path, "S1")

    stacked_dir = os.path.join(S1_DIR, task_id, "stacked")
    os.makedirs(stacked_dir, exist_ok=True)
    for base, band_paths in groups.items():
        out_path = stack_group(base, band_paths, stacked_dir, "S1")
        print("Created:", out_path)

    # Rename stacked tif files of S1 like format  S1_YYYY-MM-DD.tif
    renamed_files = []
    for fname in os.listdir(stacked_dir):
        if fname.endswith('.tif'):
             # Nếu đã đúng format chuẩn thì chỉ move sang s1_dir
            if fname.startswith("S1_") and len(fname) == len("S1_YYYY-MM-DD.tif"):
                old_path = os.path.join(stacked_dir, fname)
                new_path = os.path.join(S1_DIR, fname)
                os.rename(old_path, new_path)
                renamed_files.append(new_path)
                print(f"Moved (already standard): {fname}")
            else:
                date_str = extract_dates_from_filename(fname)
                if date_str:
                    new_name = f"S1_{date_str}.tif"
                    old_path = os.path.join(stacked_dir, fname)
                    new_path = os.path.join(stacked_dir, new_name)
                    os.rename(old_path, new_path)
                    renamed_files.append(new_path)
                    print(f"Renamed+Moved: {fname} -> {new_name}")

    # Resample 100m từ stacked -> S1_DIR
    for in_path in renamed_files:
        fname = os.path.basename(in_path)
        out_path = os.path.join(S1_DIR, fname)
        resample_s1_tifs_to_100m(in_path, out_path)
    
    # Cleanup
    shutil.rmtree(extract_path)
    

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
    Code sẽ download một file zip đặt tên theo task id, 
    Sau khi unzip, folder này chứa tất cả các ảnh tif cần download được chia ra theo từng band, 
    (cùng một file zip và metadata jsson, ta chỉ cần quan tâm đến các ảnh tif)
    """
    task_id = "4fad9bce-0ab5-4dbe-a178-2e88b1b7f259" # task id mô phỏng
    download_task(task_id, SERVER_IP, server_port=8000, save_dir=NDVI_dir)
    # Tải về xong sẽ thu được thư mục tif và các file ảnh tif được extract ra từ các file zip => xóa file zip
    zip_path = os.path.join(NDVI_dir, f"{task_id}.zip")
    extract_path = unzip_and_cleanup(NDVI_dir, task_id)
    # Ảnh NDVI chỉ có 1 band duy nhất nên bỏ qua bước stack
    # Rename tif files of S1 like format  NDVI_YYYY-MM-DD.tif
    for fname in os.listdir(extract_path):
        if fname.endswith('.tif'):
            date_str = extract_dates_from_filename(fname)
            if date_str:
                new_name = f"NDVI_{date_str}.tif"
                os.rename(os.path.join(extract_path, fname), os.path.join(extract_path, new_name))
    
    shutil.rmtree(extract_path)

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
    Code sẽ download một file zip đặt tên theo task id, 
    Sau khi unzip, folder này chứa tất cả các ảnh tif cần download được chia ra theo từng band, 
    (cùng một file zip và metadata jsson, chỉ cần quan tâm đến các ảnh)
    """
    task_id = "6cf67b64-64d0-4e51-b304-366e09210f48" # task id mô phỏng
    download_task(task_id, SERVER_IP, server_port=8000, save_dir=weather_dir)
    # Tải về xong sẽ thu được thư mục tif và các file ảnh tif được extract ra từ các file zip => xóa file zip
    zip_path = os.path.join(weather_dir, f"{task_id}.zip")
    extract_path = unzip_and_cleanup(weather_dir, task_id)

    # Stack band
    groups = group_bands(weather_dir, "Weather")
    print(groups)

    stacked_dir = os.path.join(weather_dir, task_id, "stacked")
    os.makedirs(stacked_dir, exist_ok=True)

    for base, band_paths in groups.items():
        out_path = stack_group(base, band_paths, stacked_dir, "Weather")
        print("Created:", out_path)

    # Rename tif files of weather data like format Weather_YYYY-MM-DD.tif và chuyển sang thư mục weather_dir
    for fname in os.listdir(stacked_dir):
        if fname.endswith('.tif'):
            # nếu tên đã chuẩn rồi thì bỏ qua bước đổi tên, chuyển file sang weather_dir
            if fname.startswith("Weather_") and len(fname) == len("Weather_YYYY-MM-DD.tif"):
                os.rename(
                    os.path.join(stacked_dir, fname),
                    os.path.join(weather_dir, fname)
                )
                print(f"Moved (already standard): {fname}")
            else:
                date_str = extract_dates_from_filename(fname) 
                if date_str:
                    new_name = f"Weather_{date_str}.tif"
                    os.rename(
                        os.path.join(stacked_dir, fname),
                        os.path.join(weather_dir, new_name)
                    )
                    print(f"Renamed+Moved: {fname} -> {new_name}")
    shutil.rmtree(extract_path)
    print("✅ Weather data processed:", stacked_dir)

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

    """
    Thêm
    Code for downloading soil grid and DEM images, stacking  bands and save in their respective directories
    Code sẽ download một file zip đặt tên theo task id, 
    Sau khi unzip, folder này chứa tất cả các ảnh tif cần download được chia ra theo từng band, 
    (cùng một file zip và metadata jsson, chỉ cần quan tâm đến các ảnh)
    """
    soil_task_id = "92e3a84e-e383-491a-9a67-7365f4675e80" # task id mô phỏng
    download_task(soil_task_id, SERVER_IP, server_port=8000, save_dir=soilgrid_dir)
    zip_path = os.path.join(soilgrid_dir, f"{soil_task_id}.zip")
    soil_extract_path = unzip_and_cleanup(soilgrid_dir, soil_task_id)

    soil_groups = group_bands(soil_extract_path, "SoilGrid")
    soil_stacked_dir = os.path.join(soilgrid_dir, soil_task_id,"stacked")
    os.makedirs(soil_stacked_dir, exist_ok=True)

    for base, band_paths in soil_groups.items():
        out_path = stack_group(base, band_paths, soil_stacked_dir, "SoilGrid")
        print("Created:", out_path)

    # Rename the soil grid file, chỉ đổi tên file đầu tiên tìm thấy và dừng lại
    for fname in os.listdir(soil_stacked_dir):
        if fname.endswith('.tif'):
            src = os.path.join(soil_stacked_dir, fname)

            # Nếu chạy thực thế thì cần đổi tên
            dst = os.path.join(soilgrid_dir, "SoilGrid_sand_clay_bdob.tif")

            # Chạy với dữ liệu mẫu đã có sẵn tên theo format
            dst = os.path.join(soilgrid_dir, fname)
            if src != dst:
                os.rename(src, dst)
    shutil.rmtree(soil_extract_path)

    # -------------------
    # DEM
    # -------------------
    payload_dem = {
        "collection": ["NASA/DEM"],
        "geometry": polygon_grid,
        "bands": ["elevation", "slope", "aspect"]
    }

    dem_task_id = "ec228eb5-7c13-4414-bfa9-9a6ec27c91d8" # task id mô phỏng
    download_task(dem_task_id, SERVER_IP, server_port=8000, save_dir=dem_dir)
    zip_path = os.path.join(dem_dir, f"{dem_task_id}.zip")
    dem_extract_path = unzip_and_cleanup(dem_dir, dem_task_id)

    dem_groups = group_bands(dem_extract_path, "DEM")
    print(dem_groups)
    dem_stacked_dir = os.path.join(dem_dir, dem_task_id,"stacked")
    os.makedirs(dem_stacked_dir, exist_ok=True)

    for base, band_paths in dem_groups.items():
        out_path = stack_group(base, band_paths, dem_stacked_dir, "DEM")
        print("Created:", out_path)


    # Rename the DEM file, there is only one file
    for fname in os.listdir(dem_stacked_dir):
        if fname.endswith('.tif'):
            src = os.path.join(dem_stacked_dir, fname)
            
            # Nếu chạy thực tế cần đổi tên
            # dst = os.path.join(dem_dir, "DEM_elevation_slope_aspect_30m.tif")
            
            # Chạy với dữ liệu mẫu đã có sẵn tên theo format
            dst = os.path.join(dem_dir, fname)
            if src != dst:
                os.rename(src, dst)
            break  # Chỉ đổi tên một file duy nhất
    shutil.rmtree(dem_extract_path)
    
    print("✅ SoilGrid processed:", soil_stacked_dir)
    print("✅ DEM processed:", dem_stacked_dir)

def resample_s1_tifs_to_100m(in_path, out_path):
    """
    Resample một GeoTIFF Sentinel-1 từ 10m → 100m 
    bằng cách giảm kích thước width/height theo tỉ lệ 10.
    """
    with rasterio.open(in_path) as src:
        print("Input:", in_path)
        factor = 10  # từ 10m → 100m

        new_width = src.width // factor
        new_height = src.height // factor

        # đọc và resample
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # tính lại transform cho đúng
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform
        })

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

    print(f"Resampled: {out_path}")

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


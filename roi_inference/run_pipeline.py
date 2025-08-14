import subprocess 
import os
import geopandas as gpd
from utils_data import download_image
import ee 
import geemap
from shapely.geometry import mapping
import argparse
ee.Initialize()

# Function to call download_data.py
def download_data(region, start_date, end_date):
    subprocess.run(["python", "roi_inference/download_data.py",
                    "--start_date", start_date,
                    "--end_date", end_date,
                    "--roi_path", f"roi_inference/regions_data_results/{region}/roi_inference.tif", 
                    "--region", region], check = True)

# Function to call extract_data.py
def extract_data(region):
    subprocess.run(["python", "roi_inference/extract_data.py",
                    "--region", region], check = True)

# Function to call merge_and_process.py
def process_data(region):
    subprocess.run(["python", "roi_inference/merge_and_process.py",
                    "--region", region], check = True)

# Function to call inference_emsemble_roi.py
def run_inference(region):
    subprocess.run(["python", "inference_emsemble_roi.py",
                    "--region", region,], check = True)

# Function to call prediction_visualize.py
def visualize(region):
    subprocess.run(["python", "roi_inference/prediction_visualize.py",
                    "--region", region], check = True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)

    # Các cờ điều khiển từng bước trong pipeline
    parser.add_argument("--download", action="store_true", help="Download raw data")
    parser.add_argument("--extract", action="store_true", help="Extract data")
    parser.add_argument("--process", action="store_true", help="Process + merge data")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")

    args = parser.parse_args()
    

    region = args.region
    folder = f'roi_inference/regions_data_results/{region}'
    os.makedirs(folder, exist_ok=True)
    start_date = args.start_date
    end_date = args.end_date
    
    if not os.path.exists(f"roi_inference/regions_data_results/{region}/roi_inference.tif"):
        # Download data from Sentinel-1, using any time range as long as it covers all the region
        START = '2023-01-01'
        END = '2023-12-31'

        """IF YOU WANT TO GET GEOMETRY FROM YOUR SHAPEFILE"""
        # Read shapefile
        gdf = gpd.read_file("roi_inference/regions_data_results/ngocnhat3/gialai_33_polygon/gialia_33_polygon.shp")  # hoặc .geojson

        # Select the first shape (it is the desired ROI)
        selected_shape = gdf.iloc[0].geometry

        # Convert to GeoDataFrame
        roi_gdf = gpd.GeoDataFrame(geometry=[selected_shape], crs=gdf.crs)

        # Tạo ee.FeatureCollection
        roi_ee = geemap.geopandas_to_ee(roi_gdf)

        # Get geometry of the ROI
        roi_geometry = roi_ee.geometry()


        """IF YOU WANT TO GET GEOMETRY FROM A GRID FILE, PASS IT THE PHIENHIEU OF THE GRID CELL """
        # grid_gdf = gpd.read_file('roi_inference/Grid_50K_MatchedDates.geojson')

        # selected_cell = grid_gdf.loc[grid_gdf["PhienHieu"] == region].iloc[0]
        # # Get bounding box
        # minx, miny, maxx, maxy = selected_cell.geometry.bounds
        # print(minx, miny, maxx, maxy)

        # # Generate ee.Geometry.Rectangle from coordinates
        # roi_geometry = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

        # Get Sentinel-1 data for the ROI, we will it to determine the geometry of the region
        s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(roi_geometry) \
            .filterDate(START, END) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .select(['VV', 'VH', 'angle']) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) 
        
        # Kiểm tra số ảnh
        count = s1_collection.size().getInfo()
        if count == 0:
            raise Exception("=== No Sentinel-1 images found for this region and time.")
        # Tạo ảnh mosaic
        s1_mosaic = s1_collection.mosaic()
        # Save the mosaic to a GeoTIFF file, we use it as a reference to download the data afterward
        download_image(s1_mosaic, roi_geometry, folder, f'roi_inference', 100)
    
    else: 
        print('Already downloaded roi_inference.tif')
    
    # Run the pipeline steps based on the provided flags
    if args.download:
        download_data(region, start_date, end_date)

    if args.extract:
        extract_data(region)

    if args.process:
        process_data(region)

    if args.inference:
        run_inference(region)
    
    if args.visualize:
        visualize(region)

"""
The program run pipeline to collection soil moisture data 100m from Planet as ground truth in India and China
"""
import collect_soil_moisture.s1_available_dates as get_s1_dates
import collect_soil_moisture.select_points as select_points
import collect_soil_moisture.data_100m.extract_swc_from_tiff as extract_sm 
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import os

def get_polygon_coords_from_gpkg(gpkg_path, layer=0):
    """
    Reads the first polygon from a GPKG file and returns its coordinates
    in the format required by ee.Geometry.Polygon:
    [[ [lon1, lat1], [lon2, lat2], ..., [lon1, lat1] ]]
    """
    gdf = gpd.read_file(gpkg_path, layer=layer)
    # Convert to EPSG:4326 if not already 
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print('Convert the grid into CRS:4326')
        gdf = gdf.to_crs(epsg=4326)

    # Get bounding box for the entire grid
    xmin, ymin, xmax, ymax = gdf.total_bounds
    bbox_polygon = box(xmin, ymin, xmax, ymax)

    # Extract coordinates in the required format
    coords = list(bbox_polygon.exterior.coords)
    # Convert to list of [lon, lat]
    polygon_coords = [[list(coord) for coord in coords]]
    
    return polygon_coords

def run_pipeline_100m(region, root_path, grid_path, landcover_path, tif_folder, start_date='2021-01-01', end_date='2022-12-31'):
    """
    Collect soil moisture data at 100m resolution from Planet 
    for a specified region (India or China).

    Input:  
    - region: 'india' or 'china'
    - root_path: Path to the directory of 100m data 
    - grid_path: Path to the grid of the region (gpkg file)
    - landcover_path: Path to the tif image of landcover in the region
    - tif_folder: Folder contains soil moisture tif images downloaded from Planet Variables
    - start_date: Start date for data collection (default is '2021-01-01')
    - end_date: End date for data collection (default is '2022-12-31')

    This function performs the following steps:
    1. Reads polygon (cover the region) coordinates from the grid file.        
    2. Filters the grid based on land cover data.
    3. Selects random points in the filtered grid for the specified region.
    4. Gets Sentinel-1 dates for the polygon of the region.
    5. Extracts soil moisture values from TIFF files for the selected points.
    6. Merges all soil moisture CSV files into a single CSV file.
    """

    print("*****Starting 100m soil moisture collection process...")
    # Get coordinates of the polygon that covers the region from the grid file (gpkg)
    print("*****Reading polygon coordinates from grid...")
    polygon_coords = get_polygon_coords_from_gpkg(grid_path)
    print("Coordinates of the polygon cover the region: ", polygon_coords)
    print(f"Polygon coordinates obtained: {len(polygon_coords[0])} points")

    # Filter grid based on land cover data
    print("*****Filtering grid based on land cover data...")
    threshold = 50 #The percent that sum of selected classes (crop, tree,...) over total area
    filtered_grid_path = f"{root_path}/{region}/grid/filtered_grid/tree_grass_crops.gpkg" # Name after the classes we want to keep
    filtered_grid = select_points.filter_grid(grid_path, landcover_path, threshold, filtered_grid_path)

    # Choose random points in the filtered grid for India and China
    print("*****Selecting random points for region in the filtered grid:", region)
    # Ensure filtered_grid has CRS:4326
    filtered_grid = filtered_grid.to_crs(epsg=4326)
    # Path to save csv file including selected points' information
    points_df_path = f"{root_path}/{region}/random_points_in_filtered_grid.csv"
    if region in  ['india','thaibinh']:
        points_df = select_points.choose_india_points(filtered_grid, points_df_path)
    elif region == 'china':
        points_df = select_points.choose_china_points(filtered_grid, landcover_path, points_df_path)
    else:
        raise ValueError("Region must be either 'india' or 'china'")

    # Get Sentinel-1 dates for the region
    print(f"*****Get Sentinel-1 dates for polygon of {region} from {start_date} to {end_date}")
    s1_dates_output_path = f'{root_path}/{region}/{region}_s1_metadata.csv'
    get_s1_dates.get_s1_dates(polygon_coords, start_date, end_date, s1_dates_output_path)

    # Extract soil moisture values from TIFF files
    print(f"*****Extract soil moisture from the points in {region} from {start_date} to {end_date}*****")
    # Path to save the site info
    site_info_path = f'{root_path}/{region}/{region}_site_info.csv'
    # Folder to save soil moisture information for each points
    sm_csv_folder = f'{root_path}/{region}/{region}_csv'
    os.makedirs(sm_csv_folder, exist_ok=True)
    network = region.upper()+'_100m'
    extract_sm.extract_data_for_region(network, points_df_path, tif_folder, s1_dates_output_path, site_info_path, sm_csv_folder)
    print("Saved site (point) information and soil moisture with dates for each site")


"""
Define:
region: Where we want to get soil moisture data: 'china' or 'india'
root_path : Path to the directory of 100m data
landcover_path: Path to the tif image of landcover in the region 
grid_path : Path to the grid of the region
tiff_folder: Folder contains soil moisture tif images downloaded from Planet Variables
"""

region = 'china' # or 'china'
root_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/100m'
# Create some fixed folder 
os.makedirs(f"{root_path}/{region}/{region}_csv", exist_ok=True)
os.makedirs(f"{root_path}/{region}/grid", exist_ok=True)

# Input files ussually in 'grid' folder
landcover_path = f'{root_path}/{region}/grid/dw_{region}.tif'
grid_path = f'{root_path}/{region}/grid/{region}_grid.gpkg'
tif_folder = f'{root_path}/{region}/{region}_tif'
start_date = '2021-01-01'
end_date = '2022-12-31'

if __name__ == "__main__":
    run_pipeline_100m(region, root_path, grid_path, landcover_path, tif_folder, start_date, end_date)



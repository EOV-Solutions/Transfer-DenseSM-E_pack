"""
The program run pipeline to collection soil moisture data 1km from NSIDC as ground truth 
Step 1: Split Vietnam as a 90k grid (merged from 10k grid). Filter and keep grid cells that contains points (sample.csv) where we will get sm. 
Step 2: Get Sentinel-1 date information on the filtered grid cells. 
Step 3: Determine points present on each grid cells and assign the grid cell's S1 date for those points. 
Step 4: From points where we will get data, extract soil moisture values from NSIDC tiff images. Get sm data in each in two year (2021-2022)
Step 5: With dates from Step 3 and sm values from Step 4, filter and only keep sm values that have the same date as S1 or 1 day after S1. 
Step 6: Save CSV files contains of information of all points, and filtered sm values of each point (in this we call a point as a site)
"""

import collect_soil_moisture.vn_1km_data.get_s1_dates_vn as get_s1_dates
import collect_soil_moisture.vn_1km_data.extract_swc_from_tif as extract_sm
import collect_soil_moisture.vn_1km_data.filter_swc_by_s1 as filter_sm 
import logging
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import os
import ee

ee.Initialize()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_90k_grid_from_10k(grid_path, points_csv_path, output_path):
    """
    Create a 90k grid from a 10k grid by grouping 3x3 blocks of 10k cells.
    Then filter the grid to keep only those containing points from a training_data/1km_vn/sample.csv file.
    The resulting grid will be saved to a GeoPackage file.

    Input:
    - grid_path: GeoPackage file containing the 10k grid.
    - point_csv_path: CSV file containing points with 'lon' and 'lat' columns.  
    Output:
    - grid_90km_with_points.gpkg: GeoPackage file containing the filtered grid cells.
    """
    # Load 10k grid, ensure that it has CRS EPSG:4326
    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")

    # Create 'row' and 'col' if not available in the grid
    if 'row' not in grid.columns or 'col' not in grid.columns:
        grid['centroid_x'] = grid.centroid.x.round(4)
        grid['centroid_y'] = grid.centroid.y.round(4)
        grid['row'] = grid['centroid_y'].rank(method='dense').astype(int)
        grid['col'] = grid['centroid_x'].rank(method='dense').astype(int)

    # Merge 10k grid into 90k grid by grouping into 3x3 blocks
    # Assign group id property for each 3x3 block (9 cells)
    grid['group_id'] = ((grid['row'] // 3).astype(int)).astype(str) + '_' + ((grid['col'] // 3).astype(int)).astype(str)

    # Dissolve by group_id
    merged_grid = grid.dissolve(by='group_id', as_index=False)

    # Keep only geometry and group_id properties
    merged_grid = merged_grid[['group_id', 'geometry']]

    # Assign new sequential IDs starting from 1 for simplicity 
    merged_grid = merged_grid.reset_index(drop=True)
    merged_grid['id'] = range(1, len(merged_grid) + 1)

    """ Now filtered 90k grids, keep only those containing points from training_data/1km_vn/csv/sample.csv """
    # Load points from CSV and create GeoDataFrame
    points_df = pd.read_csv(points_csv_path)
    geometry = [Point(xy) for xy in zip(points_df['lon'], points_df['lat'])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")

    # Filter merged grid to keep only those containing points
    joined = gpd.sjoin(merged_grid, points_gdf, how="inner", predicate="contains")
    selected_grid = merged_grid[merged_grid['group_id'].isin(joined['group_id'])].copy()
    # Copy 'id' column with name 'grid_id'
    selected_grid['grid_id'] = selected_grid['id']

    # Save the selected grid to a new GeoPackage file
    selected_grid.to_file(output_path, driver="GPKG")
    print(f"Saved 90k grid in {output_path}")


def run_pipeline_vn(root_path, grid_path_90k, points_csv_path, start_date, end_date ,tif_folder, network = 'VN'):
    """
    Run the pipeline to collect soil moisture data from NSIDC for Vietnam.
    Input:
    - root_path: Root path for the directory contain all data (input, output, processed).
    - grid_path_90k: Path to the 90k grid GeoPackage file.
    - points_csv_path: Path to the CSV file containing points to get soil moisture data.
    - start_date: Start date for the soil moisture data extraction.
    - end_date: End date for the soil moisture data extraction.
    - tif_folder: Folder containing NSIDC soil moisture TIFF images.
    - network: Name of the dataset (default is "VN").
    
    Steps:
    1. Get Sentinel-1 dates for the grid cells.
    2. Get Sentinel-1 dates for the points from grid cells' dates.
    3. Extract soil moisture values from NSIDC TIFF images.
    4. Filter soil moisture values based on Sentinel-1 dates.
       Save site information and individual CSV files for each station.
    5. Merge all filtered soil moisture CSV files into a single CSV file."""

    print("*****Get Sentinel-1 dates for the grid cells*****")
    # Define output directory for S1 dates fo each grid cell
    s1_dates_grid_dir = f"{root_path}/s1_dates_per_grid" # Folder containing csv files of s1 dates for each grid cell
    os.makedirs(s1_dates_grid_dir, exist_ok=True)
    get_s1_dates.get_grid_s1_dates_vn(grid_path_90k, s1_dates_grid_dir, start_date, end_date)
    print("Saved S1 dates for grid cells in", s1_dates_grid_dir)

    print("*****Get Sentinel-1 dates for the points*****")
    s1_dates_points_dir = f"{root_path}/points_s1_dates_csv_temp" # Folder containing csv files of s1 dates for each point 
    os.makedirs(s1_dates_points_dir, exist_ok=True)
    get_s1_dates.get_point_s1_dates_vn(root_path, points_csv_path, grid_path_90k, s1_dates_grid_dir, s1_dates_points_dir)
    print("Saved S1 dates for points in", s1_dates_points_dir)

    print("*****Extract soil moisture from TIF and save in CSV files*****")
    site_info_path = f'{root_path}/site_info.csv' # File CSV contaning information's sites (points) 
    sm_csv_folder = f'{root_path}/vn_sm_csv' # Folder contains CSV files of soil moisture data for each site (point)
    os.makedirs(sm_csv_folder, exist_ok=True)
    extract_sm.extract_and_create_files(points_csv_path, site_info_path, sm_csv_folder, tif_folder, network)
    print("Saved soil moisture data for each point in CSV files in", sm_csv_folder)

    # After extracting soil moisture values, we need to filter them by using Sentinel-1 dates
    # Then rewrite on csv files on sm_csv_folder
    print("*****Filter soil moisture based Sentinel-1 dates*****")
    filter_sm.filter_sm(sm_csv_folder, s1_dates_points_dir, site_info_path, network)
    print("Saved soil filtered moisture data by S1 dates for each point in CSV files in", sm_csv_folder)

root_path = "/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_vn" # Path to the folder of Vietnam soil moisture data
points_csv_path = f"{root_path}/csv/sample.csv" # CSV file contain points to get sm
# Grid of 10k and 90k of Vietnam 
grid_path_10k = f"{root_path}/grid/Grid_10K/grid_10km.gpkg"
grid_path_90k = f"{root_path}/grid/grid_90km_with_points.gpkg"
tif_folder = '/mnt/data2tb/nsidc_images' # Folder contains NSIDC tif images - aka: 1km soil moisture
network = "VN" # Network (name for the data)
# Define time range to extract soil moisture data (must have downloaded NSIDC data in this time range)
start_date = "2021-01-01"
end_date = "2022-12-31"

if __name__ == "__main__":
    # Do not need to run this step again if the 90k grid already exists
    if not os.path.exists(grid_path_90k):
        print("*****Merge 10k grid to obtain 90k grid*****")
        # There would be some warning, it's okay, dont need to be fixed
        create_90k_grid_from_10k(grid_path_10k, points_csv_path, grid_path_90k)
        
    run_pipeline_vn(
        root_path=root_path,
        grid_path_90k=grid_path_90k,
        points_csv_path=points_csv_path,
        start_date=start_date,
        end_date=end_date,
        tif_folder = tif_folder,
        network = network
    )

"""
The program run pipeline to collection soil moisture data 1km from NSIDC as ground truth 
Step 1: Split Vietnam as a 40k grid (merged from 10k grid). Filter and keep grid cells that contains points (sample.csv) where we will get sm. 
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_40k_grid_from_10k(grid_path, points_csv_path, output_path):
    """
    Create a 40k grid from a 10k grid by grouping 2x2 blocks of 10k cells.
    Then filter the grid to keep only those containing points from a training_data/1km_vn/sample.csv file.
    The resulting grid will be saved to a GeoPackage file.

    Input:
    - grid_10km.gpkg: GeoPackage file containing the 10k grid.
    - sample.csv: CSV file containing points with 'lon' and 'lat' columns.  
    Output:
    - grid_40km_with_points_1.gpkg: GeoPackage file containing the filtered grid cells.
    """

    # Load 10k grid, ensure that it has CRS EPSG:4326
    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")

    # Create 'row' and 'col' if not available in the grid
    if 'row' not in grid.columns or 'col' not in grid.columns:
        grid['centroid_x'] = grid.centroid.x.round(4)
        grid['centroid_y'] = grid.centroid.y.round(4)
        grid['row'] = grid['centroid_y'].rank(method='dense').astype(int)
        grid['col'] = grid['centroid_x'].rank(method='dense').astype(int)

    # Merge 10k grid into 40k grid by grouping into 2x2 blocks
    # Assign group id property for each 2x2 block (4 cells)
    grid['group_id'] = ((grid['row'] // 3).astype(int)).astype(str) + '_' + ((grid['col'] // 3).astype(int)).astype(str)

    # Dissolve by group_id
    merged_grid = grid.dissolve(by='group_id', as_index=False)

    # Keep only geometry and group_id properties
    merged_grid = merged_grid[['group_id', 'geometry']]

    # Assign new sequential IDs starting from 1 for simplicity 
    merged_grid = merged_grid.reset_index(drop=True)
    merged_grid['id'] = range(1, len(merged_grid) + 1)

    """ Now filtered 40k grids, keep only those containing points from training_data/1km_vn/csv/sample.csv """
    # Load points from CSV and create GeoDataFrame
    points_df = pd.read_csv(points_csv_path)
    geometry = [Point(xy) for xy in zip(points_df['lon'], points_df['lat'])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")

    # Filter merged grid to keep only those containing points
    joined = gpd.sjoin(merged_grid, points_gdf, how="inner", predicate="contains")
    selected_grid = merged_grid[merged_grid['group_id'].isin(joined['group_id'])]
    # Copy 'id' column with name 'grid_id'
    selected_grid['grid_id'] = selected_grid['id']

    # Save the selected grid to a new GeoPackage file
    selected_grid.to_file(output_path, driver="GPKG")
    print(f"Saved 40k grid in {output_path}")

def run_pipeline(root_path, grid_path_10k, points_csv_path, start_date, end_date ,tif_folder, filter_threshold=0.8):
    print("*****Merge 10k grid to obtain 40k grid")
    grid_path_40k = f"{root_path}/grid/grid_40km_with_points.gpkg"
    create_40k_grid_from_10k(grid_path_10k, points_csv_path, grid_path_40k)

    print("*****Get Sentinel-1 dates for the grid cells")
    get_s1_dates.get_grid_s1_dates_vn(root_path, grid_path_40k, start_date, end_date)

    print("*****Get Sentinel-1 dates for the points")
    get_s1_dates.get_point_s1_dates_vn(root_path)

    print("*****Extract SWC from TIF and save in CSV files")
    extract_sm.create_files_for_region()

root_path = "/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_vn"
grid_path_10k = f"{root_path}/grid/Grid_10K/grid_10km.gpkg"
points_csv_path = f"{root_path}/csv/samples"

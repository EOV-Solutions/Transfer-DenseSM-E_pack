import collect_soil_moisture.s1_available_dates as get_s1_dates
import collect_soil_moisture.select_points as select_points
import collect_soil_moisture.global_1km_data.extract_swc_from_tif as extract_sm
import collect_soil_moisture.global_1km_data.filter_swc_by_s1 as filter_sm
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

def run_pipeline_global_1km(region, root_path, grid_path, landcover_path, tiff_folder, start_date='2021-01-01', end_date='2022-12-31'):
    """
    Run the pipeline to collect soil moisture data at 1km resolution globally.
    
    Input:
    - gpkg_path: Path to the GPKG file containing the polygon of interest.
    - sm_csv_folder: Folder to save individual CSV files of soil moisture data.
    - output_path: Path to save the merged CSV file.
    - network: Name of the dataset (default is "VN").
    """
    print("*****Starting 100m soil moisture collection process...")

    # Get coordinates of the polygon that covers the region from the grid file (gpkg)
    print("*****Reading polygon coordinates from grid...")
    polygon_coords = get_polygon_coords_from_gpkg(grid_path)
    print("Coordinates of the polygon cover the region: ", polygon_coords)
    print(f"Polygon coordinates obtained: {len(polygon_coords[0])} points")

    # Filter grid based on land cover data
    print("*****Filtering grid based on land cover data...")
    threshold = 50# Theshold is the percent that sum of selected classes (crop, tree,...) over total area
    filtered_grid_path = f"{root_path}/{region}/grid/filtered_grid/tree_grass_crops.gpkg" # Name after the classes we want to keep
    filtered_grid = select_points.filter_grid(grid_path, landcover_path, threshold, filtered_grid_path)

     # Choose random points in the filtered grid for India and China
    print("*****Selecting random points for region in the filtered grid:", region)
    filtered_grid = filtered_grid.to_crs(epsg=4326) # Ensure filtered_grid has CRS:4326
    points_csv_path = f"{root_path}/{region}/random_points_in_filtered_grid.csv" # Path to save csv file including selected points' information
    if region in  ['india','thaibinh']:
        points_df = select_points.choose_india_points(filtered_grid, points_csv_path)
    elif region == 'china':
        points_df = select_points.choose_china_points(filtered_grid, landcover_path, points_csv_path)
    else:
        raise ValueError("Region must be either 'india' or 'china'")
    
     # Get Sentinel-1 dates for the region
    print(f"*****Get Sentinel-1 dates for polygon of {region} from {start_date} to {end_date}")
    s1_dates_output_path = f'{root_path}/{region}/{region}_s1_metadata_temp.csv'
    get_s1_dates.get_s1_dates(polygon_coords, start_date, end_date, s1_dates_output_path)

    # Extract soil moisture values from TIFF files
    print(f"*****Extract soil moisture from the points in {region} from {start_date} to {end_date}*****")
    # Path to save the site info
    site_info_path = f'{root_path}/{region}/{region}_site_info_temp.csv'
    # Folder to save soil moisture information for each points
    sm_csv_folder = f'{root_path}/{region}/{region}_csv_temp'
    os.makedirs(sm_csv_folder, exist_ok=True)
    network = region.upper()+'_1km'
    extract_sm.extract_and_create_files(points_csv_path, tiff_folder, site_info_path, sm_csv_folder, network)
    print("Saved site (point) information and soil moisture with dates for each site")

    # After extracting soil moisture values, we need to filter them by using Sentinel-1 dates
    # Then rewrite on csv files on sm_csv_folder
    print("*****Filter soil moisture based Sentinel-1 dates*****")
    filter_sm.filter_sm(sm_csv_folder, s1_dates_output_path, site_info_path, network)
    print("Saved soil filtered moisture data by S1 dates for each point in CSV files in", sm_csv_folder)

region = 'india' # or 'china'
root_path = '/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_global'
# Create some fixed folder 
os.makedirs(f"{root_path}/{region}/{region}_csv", exist_ok=True)
os.makedirs(f"{root_path}/{region}/grid", exist_ok=True)

# Input files ussually in 'grid' folder
landcover_path = f'{root_path}/{region}/grid/dw_{region}_1km.tif' # Image has landcover data on the region
grid_path = f'{root_path}/{region}/grid/{region}_grid.gpkg' # Grid split the region
tiff_folder = '/mnt/data2tb/nsidc_images' # Folder contains NSIDC tif images - aka: 1km soil moisture
# Time range when we get soil moisture, defined based on soil moisture data we downloaded
start_date = '2021-01-01'
end_date = '2022-12-31'

if __name__ == "__main__":
    print("*****Start collecting soil moisture data at 1km resolution globally*****")
    run_pipeline_global_1km(region, root_path, grid_path, landcover_path, tiff_folder, start_date, end_date)


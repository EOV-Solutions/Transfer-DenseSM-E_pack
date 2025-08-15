"""
Phương pháp chọn điểm lấy dữ liệu trên Việt Nam sẽ là chia Việt Nam thành các ô lưới 40k (được tạo ra từ lưới 10k), lọc và giữ lại các ô chứa vị trí 
lấy dữ liệu xác định trong file csv/sample.csv. 


Chương trình sau đây sẽ:
- Ghép tạo lưới 40k từ lưới 10k, 
- Lọc và chỉ giữ lại các ô có chứa các điểm trong csv/sample.csv
- Lưu kết quả vào file grid_40km_with_points_1.gpkg
- Từ các ô lưới 40k, sẽ xác định các ngày có dữ liệu Sentinel-1 của từng ô. 
- Lưu kết quả vào thư mục s1_dates_per_grid, mỗi ô sẽ có một file json chứa các ngày có dữ liệu Sentinel-1.
- 

Như vậy để sau này khi xử lý giá trị sm 1km NSIDC của các điểm 
  thuộc các ô lưới 40k, ta sẽ chỉ giữ lại các giá trị sm có ngày có dữ liệu Sentinel-1.

  Đầu ra sẽ là các file csv chứa ngày có dữ liệu Sentinel-1 của từng ô trong thư mục: s1_dates_per_grid
"""
import ee
import json
import os
import time
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path

# Path to the data directory
# root_path =  "/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_vn"
# Authenticate and initialize Earth Engine
ee.Initialize()


def create_40k_grid_from_10k(root_path):
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

    # Load 10k grid
    grid = gpd.read_file(f"{root_path}/grid/Grid_10K/grid_10km.gpkg").to_crs("EPSG:4326")

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
    points_df = pd.read_csv(f"{root_path}/csv/sample.csv")
    geometry = [Point(xy) for xy in zip(points_df['lon'], points_df['lat'])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")

    # Filter merged grid to keep only those containing points
    joined = gpd.sjoin(merged_grid, points_gdf, how="inner", predicate="contains")
    selected_grid = merged_grid[merged_grid['group_id'].isin(joined['group_id'])]
    # Copy 'id' column with name 'grid_id'
    selected_grid['grid_id'] = selected_grid['id']

    # Save the selected grid to a new GeoPackage file
    selected_grid.to_file(f"{root_path}/grid/grid_40km_with_points_1.gpkg", driver="GPKG")


    
def get_grid_s1_dates_vn(root_path, grid_path, start_date, end_date):
    """
    Get Sentinel-1 dates for each grid cell in the 40k grid from the defined time range.
    The results will be saved in JSON files in the training_data/1km_vn/s1_dates_per_grid directory.

    INPUT:
    - start_date: Start date in "YYYY-MM-DD" format.    
    - end_date: End date in "YYYY-MM-DD" format.
    OUTPUT:     
    - s1_dates_per_grid: Directory containing JSON files with Sentinel-1 dates for each filtered 40k grid cell.
    """

    # Get Sentinel-1 dates for each grid cell from 2021 to 2022
    # start_date = "2021-01-01"
    # end_date = "2022-12-31"

    # INPUT : filtered 40k grid file 
    grid_file = f"{root_path}/grid/grid_40km_with_points_1.gpkg"  # Must contain a 'grid_id' column
    output_dir = f"{root_path}/s1_dates_per_grid"
    os.makedirs(output_dir, exist_ok=True)

    # === Load grid geometries ===
    grid_gdf = gpd.read_file(grid_file).to_crs("EPSG:4326")

    # Function to get Sentinel-1 dates for a given geometry and date range
    def get_s1_dates(geom, start_date, end_date, orbit_pass):
        ee_geom = ee.Geometry(geom.__geo_interface__)
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterDate(start_date, end_date) \
            .filterBounds(ee_geom) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)) \
            .select(["VV", "VH"])
        
        dates = s1.aggregate_array("system:time_start").getInfo()
        unique_dates = sorted(set([
            ee.Date(d).format("YYYY-MM-dd").getInfo() for d in dates
        ]))
        # Return Sentinel-1 dates as a list of unique date strings
        return unique_dates

    # Loop through each grid cell and retrieve Sentinel-1 dates
    for _, row in grid_gdf.iterrows():
        grid_id = row["id"]
        geom = row["geometry"]

        # Check if dates already exist for this grid_id
        if os.path.exists(os.path.join(output_dir, f"s1_dates_{grid_id}.json")):
            print(f"Already retrieved {grid_id} data")
            continue

        try:
            ascending_dates = get_s1_dates(geom, start_date, end_date, "ASCENDING")
            descending_dates = get_s1_dates(geom, start_date, end_date, "DESCENDING")

            if not ascending_dates and not descending_dates:
                print(f"No Sentinel-1 data for {grid_id}")
                continue
            
            # Save the dates to a JSON file
            out_data = {
                "grid_id": grid_id,
                "ascending": ascending_dates,
                "descending": descending_dates
            }

            out_path = os.path.join(output_dir, f"s1_dates_{grid_id}.json")
            with open(out_path, "w") as f:
                json.dump(out_data, f)

            print(f"Saved dates for {grid_id}: {len(ascending_dates)} ASC, {len(descending_dates)} DESC")

            time.sleep(5)

        except Exception as e:
            print(f"Failed for {grid_id}: {e}")




"""Chúng ta đã thu thập các ngày có dữ liệu S1 trong từng ô của grid. Bước tiếp theo sẽ tìm các điểm thuộc từng ô và gán ngày S1 đã tìm được cho các điểm đó.
Đầu ra sẽ là các file csv chứa các ngày S1 tương ứng với từng điểm, lưu trong thư mục training_data/1km_vn/points_s1_dates_csv."""
def get_point_s1_dates_vn(root_path):
    """
    Assign Sentinel-1 dates to each point in the 40k grid based on the grid cells they belong to.
    The results will be saved in CSV files in the training_data/1km_vn/points_s1_dates_csv directory.
    """

    # Paths
    points_csv = f'{root_path}/csv/sample.csv'
    grid_file = f'{root_path}/grid/grid_40km_with_points_1.gpkg'
    grid_dates_folder = Path(f'{root_path}/s1_dates_per_grid')
    output_folder = Path(f'{root_path}/points_s1_dates_csv')
    output_folder.mkdir(exist_ok=True)

    # Load points data (lat, lon) from CSV
    df_points = pd.read_csv(points_csv)
    points_gdf = gpd.GeoDataFrame(df_points,
                                geometry=gpd.points_from_xy(df_points['lon'], df_points['lat']),
                                crs='EPSG:4326')

    # Load filtered 40k grid
    grid_gdf = gpd.read_file(grid_file)
    grid_gdf = grid_gdf.to_crs(points_gdf.crs)

    # Dictionary to store points in each grid cell
    grid_points = {}
    # Loop through each grid cell
    for idx, grid_row in grid_gdf.iterrows():
        grid_id = grid_row['grid_id']  # grid_id is also the name of the JSON file containing S1 dates for this grid
        grid_geom = grid_row['geometry']
        
        # Get all points within this grid cell
        points_in_grid = points_gdf[points_gdf.geometry.within(grid_geom)]
        points_list = points_in_grid['id'].tolist()
        # Store points in the grid_points dictionary
        grid_points[grid_id] = points_list

        # If no points in this grid cell, skip to next grid cell
        if points_in_grid.empty:
            continue

        # Open the corresponding date JSON file for this grid cell
        json_path = grid_dates_folder/f's1_dates_{grid_id}.json'
        if not json_path.exists():
            print(f'Not found json file for {grid_id}')
            continue
        with open(json_path, 'r') as f:
            s1_dates = json.load(f)

        # Merge ascending and descending dates into a single DataFrame
        asc_dates = pd.DataFrame({'s1_date': s1_dates.get('ascending', []), 'orbit_pass': 'ASC'})
        desc_dates = pd.DataFrame({'s1_date': s1_dates.get('descending', []), 'orbit_pass': 'DESC'})
        all_dates = pd.concat([asc_dates, desc_dates], ignore_index=True).sort_values('s1_date')

        # Traverse all points in this grid cell and assign its S1 dates to all those points. 
        # Create csv file for each point
        for _, point_row in points_in_grid.iterrows():
            point_id = point_row['id']
            output_path = output_folder / f'{point_id}.csv'
            all_dates.to_csv(output_path, index=False)

    with open(f'{root_path}/grid/grid_points.json', 'w') as f:
        json.dump(grid_points, f, ensure_ascii = False, indent = 2)





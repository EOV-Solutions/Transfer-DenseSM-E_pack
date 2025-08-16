"""
Phương pháp chọn điểm lấy dữ liệu trên Việt Nam sẽ là chia Việt Nam thành các ô lưới 90k (được tạo ra từ lưới 10k), lọc và giữ lại các ô chứa vị trí 
lấy dữ liệu xác định trong file csv/sample.csv. 


Chương trình sau đây sẽ:
- Ghép tạo lưới 90k từ lưới 10k, 
- Lọc và chỉ giữ lại các ô có chứa các điểm trong csv/sample.csv
- Lưu kết quả vào file grid_90km_with_points.gpkg
- Từ các ô lưới 90k, sẽ xác định các ngày có dữ liệu Sentinel-1 của từng ô. 
- Lưu kết quả vào thư mục s1_dates_per_grid, mỗi ô sẽ có một file json chứa các ngày có dữ liệu Sentinel-1.
- 

Như vậy để sau này khi xử lý giá trị sm 1km NSIDC của các điểm 
  thuộc các ô lưới 90k, ta sẽ chỉ giữ lại các giá trị sm có ngày có dữ liệu Sentinel-1.

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
    
def get_grid_s1_dates_vn(grid_path, output_dir, start_date, end_date):
    """
    Get Sentinel-1 dates for each grid cell in the 90k grid from the defined time range.
    The results will be saved in JSON files in the training_data/1km_vn/s1_dates_per_grid directory.

    INPUT:
    - start_date: Start date in "YYYY-MM-DD" format.    
    - end_date: End date in "YYYY-MM-DD" format.
    OUTPUT:     
    - s1_dates_per_grid: Directory containing JSON files with Sentinel-1 dates for each filtered 90k grid cell.
    """

    # Get Sentinel-1 dates for each grid cell from 2021 to 2022
    # start_date = "2021-01-01"
    # end_date = "2022-12-31"

    os.makedirs(output_dir, exist_ok=True)

    # === Load grid geometries ===, ensure that it has CRS EPSG:4326
    grid_gdf = gpd.read_file(grid_path).to_crs("EPSG:4326")
    # Ensure contain 'grid_id' column
    if 'grid_id' not in grid_gdf.columns:
        raise ValueError("Grid 90k GeoDataFrame must contain 'grid_id' column")

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

            print(f"Saved dates for grid cell {grid_id}: {len(ascending_dates)} ASC, {len(descending_dates)} DESC")

            time.sleep(5)

        except Exception as e:
            print(f"Failed for {grid_id}: {e}")




"""Chúng ta đã thu thập các ngày có dữ liệu S1 trong từng ô của grid. Bước tiếp theo sẽ tìm các điểm thuộc từng ô và gán ngày S1 đã tìm được cho các điểm đó.
Đầu ra sẽ là các file csv chứa các ngày S1 tương ứng với từng điểm, lưu trong thư mục training_data/1km_vn/points_s1_dates_csv."""
def get_point_s1_dates_vn(root_path, points_csv, grid_file, grid_dates_folder, output_folder):
    """
    Assign Sentinel-1 dates to each point in the 90k grid based on the grid cells they belong to.
    The results will be saved in CSV files in the training_data/1km_vn/points_s1_dates_csv directory.

    INPUT:
    """

    # Load points data (lat, lon) from CSV
    df_points = pd.read_csv(points_csv)
    points_gdf = gpd.GeoDataFrame(df_points,
                                geometry=gpd.points_from_xy(df_points['lon'], df_points['lat']),
                                crs='EPSG:4326')

    # Load filtered 90k grid
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
        json_path = f"{grid_dates_folder}/s1_dates_{grid_id}.json"
        if not os.path.exists(json_path):
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
            output_path = f"{output_folder}/{point_id}.csv"
            all_dates.to_csv(output_path, index=False)

    # Save this json file to check points in each grid cells (visualization in qgis)
    with open(f'{root_path}/grid/grid_points.json', 'w') as f:
        json.dump(grid_points, f, ensure_ascii = False, indent = 2)





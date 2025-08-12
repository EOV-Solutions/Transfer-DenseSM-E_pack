"""Trong file get_s1_dates.ipynb, chúng ta đã thu thập các ngày có dữ liệu S1 trong từng ô của grid.
Code này sẽ tìm các điểm thuộc từng ô và gán ngày S1 đã tìm được cho các điểm đó.
Đầu ra sẽ là các file csv chứa các ngày S1 tương ứng với từng điểm. points_s1_dates_csv."""

import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import Point
from pathlib import Path

# Đường dẫn
points_csv = '/mnt/data2tb/Transfer-DenseSM-E_2/1km_vn_data/csv/sample.csv'
grid_file = '/mnt/data2tb/Transfer-DenseSM-E_2/1km_vn_data/grid/grid_40km_with_points_1.gpkg'
json_folder = Path('/mnt/data2tb/Transfer-DenseSM-E_2/1km_vn_data/2020/s1_dates_per_grid')  # thư mục chứa grid_{grid_id}.json
output_folder = Path('/mnt/data2tb/Transfer-DenseSM-E_2/1km_vn_data/2020/points_s1_dates_csv')
output_folder.mkdir(exist_ok=True)

# Load dữ liệu điểm
df_points = pd.read_csv(points_csv)
points_gdf = gpd.GeoDataFrame(df_points,
                               geometry=gpd.points_from_xy(df_points['lon'], df_points['lat']),
                               crs='EPSG:4326')

# Load lưới
grid_gdf = gpd.read_file(grid_file)
grid_gdf = grid_gdf.to_crs(points_gdf.crs)

grid_points = {}
# Duyệt từng ô lưới
for idx, grid_row in grid_gdf.iterrows():
    grid_id = grid_row['grid_id']  # chính là tên file JSON
    grid_geom = grid_row['geometry']
    
    # Lấy các điểm nằm trong ô lưới này
    points_in_grid = points_gdf[points_gdf.geometry.within(grid_geom)]
    points_list = points_in_grid['id'].tolist()

    grid_points[grid_id] = points_list
    
    if points_in_grid.empty:
        continue

    # Mở file JSON tương ứng ô đấy
    json_path = json_folder / f's1_dates_{grid_id}.json'
    if not json_path.exists():
        print(f'Not found json file for {grid_id}')
        continue
    
    with open(json_path, 'r') as f:
        s1_dates = json.load(f)

    # Gộp ngày của cả 2 loại ASC/DESC thành DataFrame
    asc_dates = pd.DataFrame({'s1_date': s1_dates.get('ascending', []), 'orbit_pass': 'ASC'})
    desc_dates = pd.DataFrame({'s1_date': s1_dates.get('descending', []), 'orbit_pass': 'DESC'})
    all_dates = pd.concat([asc_dates, desc_dates], ignore_index=True).sort_values('s1_date')

    # Duyệt qua tất cả các điểm trong ô lưới, và gán ngày S1 của ô cho các tất cả các điểm đấy. 
    # Create csv file for each point
    for _, point_row in points_in_grid.iterrows():
        point_id = point_row['id']
        output_path = output_folder / f'{point_id}.csv'
        all_dates.to_csv(output_path, index=False)

with open('1km_vn_data/grid/grid_points.json', 'w') as f:
    json.dump(grid_points, f, ensure_ascii = False, indent = 2)

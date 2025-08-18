"""
Đối với dữ liệu soil moisture 100m, chúng ta sẽ chọn điểm ngẫu nhiên trên các vùng có dữ liệu do Planet cung cấp. Phương pháp chọn như sau:
- Chia vùng dữ liệu thành các grid 1km, lọc và giữ lại các ô có phần lớn diện tích là crop, tree và grass.
- Với mỗi ô được giữ lại, chọn một điểm ngẫu nhiên trong đó. 
"""

import ee
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.transform
import numpy as np
import pandas as pd
import shapely
import random
import traceback 
import sys

ee.Initialize()

def download_landcover_data():
    """
    Get Dynamic World V1 tif image over the ROI in Inidia and China

    Already downloaded, so no need to run again. The land cover images are stored in the following paths:
    * India: training_data/100m/india/map/dw_india.tif
    * China: training_data/100m/china/map/dw_china.tif
    """

    # Choose randon time period for the land cover data as long as it covers the ROI
    start = '2022-10-01'
    end = '2022-10-31'

    # Chú ý cần thay đổi đường dẫn đến file geojson của bạn
    roi = ee.FeatureCollection('projects/ee-lengocthanh/assets/china_grid').geometry()

    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(start, end) \
        .filterBounds(roi) \
        .select('label') \
        .mode()

    # Export as GeoTIFF
    task = ee.batch.Export.image.toDrive(
        image=dw,
        description='dw_china',
        folder='gee_exports',
        fileNamePrefix='dw_mode_2022',
        region=roi.bounds(),
        scale=10,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()


def filter_grid(grid_path, landcover_path, threshold = 50, output_path = None):

    """Filter the grid based on the land cover data
    This function reads the land cover data and filters the grid based on three preferred classes: crop, tree, land.
    If a grid cell have the percentage of the area of the selected selected over 'threshold', then keep it
    
    Input: 
     -region : 'china' or 'india'
    Output:
     - Filtered grid 
     - Save the the filtered grid based on classes (crop, grass, tree)
    """

    # Load and convert your grid to EPSG:4326
    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")

    # Open the Dynamic World image (already downloaded as GeoTIFF)
    with rasterio.open(landcover_path) as src:

        # --- Check CRS ---
        if grid.crs is None:
            sys.exit("[ERROR] Grid has no CRS defined.")
        if src.crs is None:
            sys.exit("[ERROR] Landcover raster has no CRS defined.")
        if grid.crs.to_epsg() != 4326:
            sys.exit(f"[ERROR] Grid CRS is {grid.crs}, expected EPSG:4326.")

        results = []

        for idx, row in grid.iterrows():
            # Get the geometry of the grid cell
            geom = [row['geometry']]
            try:
                # Mask the land cover raster with the grid cell geometry
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                # Land cover labels are in the first band
                label_pixels = out_image[0]
                # Remove nodata values
                valid_pixels = label_pixels[label_pixels != src.nodata]
                
                if len(valid_pixels) == 0:
                    continue

                unique, counts = np.unique(valid_pixels, return_counts=True)
                label_count = dict(zip(unique, counts))

                selected_total = sum(label_count.get(k, 0) for k in [1, 2, 4])
                total = sum(counts)

                # Calculate the percentage of selected classes
                percent = selected_total / total * 100 
                # If the percentage is greater than threshold, add the grid cell to results
                if percent > threshold:
                    results.append(row['id'])

            except Exception as e:
                print(f"[ERROR] Failed on grid cell {row['id']}: {e}")
                print("Please, check the CRS of landcover data and the grid!!!")

    # Create a new GeoDataFrame with only the selected grid cells
    selected_grid = grid[grid['id'].isin(results)]

    # Save to GPKG
    selected_grid.to_file(output_path, driver="GPKG")
    print(f"Filtered grid saved to {output_path}")
    return selected_grid

"""
Không thể dùng code chung để chọn điểm lấy dữ liệu trên vùng Trung Quốc và Ấn Độ
Bởi vì có sự đặc biệt ở vùng lấy dữ liệu tại Trung Quốc. Ở đó đa số toàn đồi núi do đó, 
nếu lấy điểm ngẫu nhiên trong các grid cell được lọc sẽ thường rơi vào các vùng là rừng núi. 
Cách giải quyết đó là, xét trong từng grid cell, nếu tại ô đó có vùng crop sẽ chọn điểm ngẫu nhiên trên crop, 
còn không có crop trong ô, thì sẽ phải bắt buộc chọn điểm ngẫu nhiên trên vùng rừng núi.
Còn với Ấn Độ, vùng lấy dữ liệu là đồng bằng nên không cần phải làm phức tạp như vậy. 
Chỉ đơn giản là chọn điểm ngẫu nhiên trong từng grid cell
"""

def choose_india_points(grid, output_path):
    """
    Choose random points in the filtered grid for India
    """
    # Read the grid from a GPKG file
    # grid = gpd.read_file("china/map/filtered_grid/tree_grass_crops.gpkg").to_crs("EPSG:4326")

    points = []
    # Traverse each grid cell in the filtered GeoDataFrame
    for idx, row in grid.iterrows():
        polygon = row['geometry']
        grid_id = row['id']
        # Generate a random point within the polygon
        minx, miny, maxx, maxy = polygon.bounds
        while True:
            random_point = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(random_point):
                break
        points.append({'id': grid_id, 'longitude': random_point.x, 'latitude': random_point.y})

    # Save to CSV
    points_df = pd.DataFrame(points)
    points_df.to_csv(output_path, index=False)
    print(f"Selected random points saved to {output_path}")
    return points_df

def choose_china_points(grid, landcover_path, output_path):
    """
    Select random locations for getting soil moisture data from filtered grid in China.
    Because Chinese region is mountainous, so that we have to choose randon locations in crop area. 
    If there is no crop in the grid cell, we will select random locations.
    """
    # grid = gpd.read_file("china/map/filtered_grid/tree_grass_crops.gpkg").to_crs("EPSG:4326")

    points = []

    # Open land cover raster
    with rasterio.open(landcover_path) as src:
        for idx, row in grid.iterrows():
            polygon = row['geometry']
            grid_id = row['id']
            # Get the geometry of the grid cell
            geom = [polygon]

            try: 
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                label_pixels = out_image[0]
                valid_mask = label_pixels != src.nodata

                # find crop pixels (label : 4)
                crop_mask = (label_pixels == 4) & valid_mask 
                
                # If there are crop pixels in the grid cell, choose a random one
                if np.any(crop_mask):
                    # Get indices of crop_pixels
                    crop_indices = np.argwhere(crop_mask)
                    # Choose a random crop pixel
                    y, x = crop_indices[random.randint(0, len(crop_indices)-1)]
                    # convert pixel indices to coordinates 
                    lon, lat = rasterio.transform.xy(out_transform, y, x)
                
                else:
                    # No crop region, pick random point in grid cell
                    minx, miny, maxx, maxy = polygon.bounds 
                    while True:
                        random_point = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                        if polygon.contains(random_point):
                            lon, lat = random_point.x, random_point.y
                            break
                points.append({'id': grid_id, 'longitude': lon, 'latitude': lat})
            
            except Exception as e:
                print(f"Failed on grid cell {grid_id} : {e}")

    # Save to CSV
    points_df = pd.DataFrame(points)
    points_df.to_csv(output_path, index=False)
    print(f"Selected random points saved to {output_path}")
    return points_df
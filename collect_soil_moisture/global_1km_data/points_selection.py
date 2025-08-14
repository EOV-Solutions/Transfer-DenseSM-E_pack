import ee
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
import shapely
import random

ee.Initialize()

root_path = "/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_global"

"""Get Dynamic World V1 tif image over the ROI
We use landCover from Dynamic World data to determine the area we want to get soil moisture. 
There are three areas that need to be concerned:
1 . tree
2 . grass
4 . crop
1, 2, 4 is the id in the Dynamic World data."""

def export_landcover_data():
    """
    Get landcover data from Dynamic World collection to select the locations to get soil Moisture data later
    """
    # Choose random range time as long as there is no missing Data
    start = '2022-10-01'
    end   = '2022-10-31'

    # Your region of interest
    # geojson_path = 'china/map/china_grid.geojson'
    # roi = geemap.geojson_to_ee(geojson_path)

    # Manually upload the ROI file on GEE and load it from Earth Engine assets
    roi = ee.FeatureCollection('projects/ee-lengocthanh/assets/1km_china_roi').geometry()

    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(start, end) \
        .filterBounds(roi) \
        .select('label') \
        .mode()

    # Export as GeoTIFF
    task = ee.batch.Export.image.toDrive(
        image=dw,
        description='dw_china_1km',
        folder='gee_exports',
        fileNamePrefix='dw_china_1km',
        region=roi.bounds(),
        scale=10,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()



# region = 'india' or 'china'
def filter_grid_cells_on_landcover(region = 'india'):
    """
    Filter and keep grid cells meeting the requirements
    Using landcover data to filter grids, keep grids that have large are of classes we want to get soil moisture(tree, grass, crop). 
    Then choose random points in filtered grid to get data.
    
    Input: 
     -region : 'china' or 'india'
    Output:
     - Save the the filtered grid based on classes (crop, grass, tree)
    """

    # Load and convert your grid to EPSG:4326
    grid = gpd.read_file(f"{root_path}/{region}/grid/2km_grid.gpkg").to_crs("EPSG:4326")

    # Open the Dynamic World image (already downloaded as GeoTIFF by function 'export_landcover_data')
    with rasterio.open(f"{root_path}/{region}/grid/dw_{region}_1km.tif") as src:
        results = []
        # Traverse each grid cell in the GeoDataFrame
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
                
                # Count frequencies of each label 
                unique, counts = np.unique(valid_pixels, return_counts=True)
                label_count = dict(zip(unique, counts))

                # Sum up pixels for 3 classes 1 (tree), 2 (grass), 4 (crop)
                selected_total = sum(label_count.get(k, 0) for k in [1,2,4])
                total = sum(counts)

                # Calculate the percentage of selected classes 
                percent = selected_total / total * 100

                # If the percentage is greater than 50%, add the grid cell to results
                if percent > 50:
                    results.append(row['id'])  # or use row.name, or any identifier

            except Exception as e:
                print(f"Failed on grid cell {row['id']}: {e}")

    # all grid cell ids
    all_ids = grid['id']

    # Create a new GeoDataFrame with only the selected grid cells (selected based on the results)
    selected_grid = grid[grid['id'].isin(results)]

    # Save to GPKG
    selected_grid.to_file(f"{root_path}/{region}/grid/filtered_grid/tree_grass_crops.gpkg", driver="GPKG")

"""
Không thể dùng code chung để chọn điểm lấy dữ liệu trên vùng Trung Quốc và Ấn Độ
Bởi vì có sự đặc biệt ở vùng lấy dữ liệu tại Trung Quốc. Ở đó đa số toàn đồi núi do đó, 
nếu lấy điểm ngẫu nhiên trong các grid cell được lọc sẽ thường rơi vào các vùng là rừng núi. 
Cách giải quyết đó là, xét trong từng grid cell, nếu tại ô đó có vùng crop sẽ chọn điểm ngẫu nhiên trên crop, 
còn không có crop trong ô, thì sẽ phải bắt buộc chọn điểm ngẫu nhiên trên vùng rừng núi.
Còn với Ấn Độ, vùng lấy dữ liệu là đồng bằng nên không cần phải làm phức tạp như vậy. 
Chỉ đơn giản là chọn điểm ngẫu nhiên trong từng grid cell
"""

def select_points_for_india():
    """
    Selection random locations for getting soil moisture data from filtered grid in India.

    Save the selected points in  'random_points_in_filtered_grid.csv' file in the folder of india.
    """
    # Read the grid from a GPKG file that includes filtered grid
    grid = gpd.read_file(f"{root_path}/india/grid/filtered_grid/tree_grass_crops.gpkg").to_crs("EPSG:4326")

    points = []
    # Traverse each grid cell in the filtered GeoDataFrame
    for idx, row in grid.iterrows():
        polygon = row['geometry']
        grid_id = row['id']
        # Generate a random point within the polygon of the grid cell
        minx, miny, maxx, maxy = polygon.bounds
        while True:
            random_point = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(random_point):
                break
        points.append({'id': grid_id, 'lon': random_point.x, 'lat': random_point.y})

    # Save to CSV
    points_df = pd.DataFrame(points)
    points_df.to_csv(f"{root_path}/india/grid/filtered_grid/random_points_in_filtered_grid.csv", index=False)

def select_points_for_china():
    """
    Selection random locations for getting soil moisture data from filtered grid in China.
    Because Chinese region is mountainous, so that we have to choose randon locations in crop area. 
    If there is no crop in the grid cell, we will select random locations.

    Save the selected points in  'random_points_in_filtered_grid.csv' file in the folder of china.
    """
    # Paths to filtered grid and land cover raster files
    grid_path = f"{root_path}/china/grid/filtered_grid/tree_grass_crop.gpkg"
    dw_tif_path = f"{root_path}/china/grid/dw_china_1km.tif"

    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")

    points = []

    # Open land cover raster
    with rasterio.open(dw_tif_path) as src:
        # Traverse each grid cell in the filtered GeoDataFrame
        for idx, row in grid.iterrows():
            polygon = row['geometry']
            grid_id = row['id']
            # Get the geometry of the grid cell
            geom = [polygon]

            try: 
                out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
                label_pixels = out_image[0]
                valid_mask = label_pixels != src.nodata

                # find crop pixels (lable : 4) mask
                crop_mask = (label_pixels == 4) & valid_mask 

                # If there are crop pixels in the grid cell, choose a random one
                if np.any(crop_mask):
                    # Get indices of crop_pixels
                    crop_indices = np.argwhere(crop_mask)
                    # Choose a random crop pixel
                    y, x = crop_indices[random.randint(0, len(crop_indices)-1)]
                    # convert pixel indices to coordinates 
                    lon, lat = rasterio.transform.xy(out_transform, y, x)
                
                # No crop region, pick random point in grid cell
                else:
                    minx, miny, maxx, maxy = polygon.bounds 
                    while True:
                        random_point = shapely.geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                        if polygon.contains(random_point):
                            lon, lat = random_point.x, random_point.y
                            break
                points.append({'id': grid_id, 'lon': lon, 'lat': lat})
            
            except Exception as e:
                print(f"Failed on grid cell {grid_id} : {e}")

    # Save to CSV
    points_df = pd.DataFrame(points)
    points_df.to_csv(f"{root_path}/china/grid/filtered_grid/random_points_in_filtered_grid.csv", index=False)    


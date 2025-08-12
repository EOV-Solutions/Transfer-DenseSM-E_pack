import os
import pandas as pd
import numpy as np
from osgeo import osr
import geopandas
import ee
import rasterio
from rasterio.transform import xy 
from shapely.geometry import Polygon, Point, box
from pyproj import Transformer
import re
import requests
from shapely.geometry import Polygon
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
# ee.Authenticate(force=True)
ee.Initialize()



class PointGeometry:
    # used for reprojection, build point with rectangle buffer
    def __init__(self, source_EPSG, target_proj):
        self.source_EPSG = source_EPSG
        self.target_proj = target_proj
        self.transform = self.build_geo_transform()

    def build_geo_transform(self):
        source = osr.SpatialReference()
        source.ImportFromEPSG(self.source_EPSG)
        target = osr.SpatialReference()
        if type(self.target_proj) == int:
            target.ImportFromEPSG(self.target_proj)
        else:
            target.ImportFromProj4(self.target_proj)
        return osr.CoordinateTransformation(source, target)

    def re_project(self, x, y):
        """
        :param x: Longitude or x
        :param y: Latitude or y
        :return:
        """
        if self.source_EPSG == 4326:
            location = self.transform.TransformPoint(y, x)
        else:
            location = self.transform.TransformPoint(x, y)
            location = [location[1], location[0]]
        return location


class grids_4_a_region:
    # used for reprojection, build point with rectangle buffer
    def __init__(self, source_EPSG, gridSize=9):
        self.source_EPSG = source_EPSG
        self.target_proj = 6933  # EASE 2.0 EPSG

        self.EASE_ulx = -17367530.45
        self.EASE_uly = 7314540.83
        if gridSize == 9:
            self.pixelWidth = 9008.05  # 36032.22
            self.pixelHeight = -9008.05  # 36032.22
        elif gridSize == 36:
            self.pixelWidth = 36032.22
            self.pixelHeight = -36032.22
        elif gridSize == 72:
            self.pixelWidth = 36032.22 * 2
            self.pixelHeight = -36032.22 * 2
        elif gridSize <= 1:
            self.pixelWidth = 1000.90*gridSize
            self.pixelHeight = -1000.90*gridSize

        self.pgeo = PointGeometry(self.source_EPSG, self.target_proj)
        self.pgeo_inv = PointGeometry(self.target_proj, self.source_EPSG)
        self.r=0
        self.c=0
        self.lon=0
        self.lat=0

    def get_boundary(self, path_2_region_shp):
        region_shp = geopandas.read_file(path_2_region_shp)
        top_left = self.pgeo.re_project(region_shp.bounds['minx'].min(), region_shp.bounds['maxy'].max())
        bottom_right = self.pgeo.re_project(region_shp.bounds['maxx'].max(), region_shp.bounds['miny'].min())
        minx, maxy = top_left[:2]
        maxx, miny = bottom_right[:2]

        minx = np.floor((minx - self.EASE_ulx) / self.pixelWidth) * self.pixelWidth + self.EASE_ulx
        maxx = np.ceil((maxx - self.EASE_ulx) / self.pixelWidth) * self.pixelWidth + self.EASE_ulx
        miny = np.ceil((miny - self.EASE_uly) / self.pixelHeight) * self.pixelHeight + self.EASE_uly
        maxy = np.floor((maxy - self.EASE_uly) / self.pixelHeight) * self.pixelHeight + self.EASE_uly

        outputBounds = (minx, maxx, miny, maxy)
        return outputBounds

    def xy_2_cr(self, x, y):
        c = []
        r = []
        for xx, yy in zip(x, y):
            c.append(int(np.floor((xx - self.EASE_ulx) / self.pixelWidth)))
            r.append(int(np.ceil((yy - self.EASE_uly) / self.pixelHeight)))
        return r, c

    def cr_2_xy(self, c, r):
        x = []
        y = []
        for cc, rr in zip(c, r):
            x.append(cc * self.pixelWidth + self.EASE_ulx)
            y.append(rr * self.pixelHeight + self.EASE_uly)
        return x, y

    def cr_gee_ring(self, c, r):
        bx = [c * self.pixelWidth + self.EASE_ulx,
              (c + 1) * self.pixelWidth + self.EASE_ulx,
              (c + 1) * self.pixelWidth + self.EASE_ulx,
              c * self.pixelWidth + self.EASE_ulx,
              c * self.pixelWidth + self.EASE_ulx]
        by = [r * self.pixelHeight + self.EASE_uly,
              r * self.pixelHeight + self.EASE_uly,
              (r - 1) * self.pixelHeight + self.EASE_uly,
              (r - 1) * self.pixelHeight + self.EASE_uly,
              r * self.pixelHeight + self.EASE_uly]
        ring = []
        for x, y in zip(bx, by):
            temp = self.pgeo_inv.re_project(x, y)
            ring.append(temp[:2])
        return ring, bx, by

    def get_wgs_grid(self, x, y):
        self.lat=y
        self.lon=x
        # Transform (x, y) coordinates into a different projection system (4326)
        targetxy = self.pgeo.re_project(x, y)
        xx, yy = targetxy[:2]
        # Convert the projected coordinates into row and column 
        r, c = self.xy_2_cr([xx], [yy])
        self.r=r
        self.c=c
        # Generate a bounding box around a grid cell defined  by the row and column indicces and reprojects the coordinates back to the original coorindate system
        ring, bx, by = self.cr_gee_ring(c[0], r[0])
        ring_target = [(bx[0], by[1]),
                       (bx[1], by[1]),
                       (bx[2], by[2]),
                       (bx[3], by[3])]
        return ring, ring_target

    def get_pixel_ring(self, x, y, res):
        res = res / 2
        polygon_ring = []
        for i in range(len(x)):
            loc = self.pgeo.re_project(x[i], y[i])
            polygon_ring.append([(loc[0] - res, loc[1] - res),
                                 (loc[0] - res, loc[1] + res),
                                 (loc[0] + res, loc[1] + res),
                                 (loc[0] + res, loc[1] - res)])
        return polygon_ring


def extract_date(file_name): 
    match = re.search(r'(\d{4}-\d{2}-\d{2})', file_name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No date found in file name: {file_name}")
    
def normalizingData(X, min_per, max_per):
    temp=(X - min_per) / (max_per - min_per)
    temp[temp>1]=1
    temp[temp<0]=0
    return temp

def get_band_values(tif_path):
    """
    Return a numpy array of shape (3, num_pixels) for bands ['VV', 'VH', 'angle'] from a GeoTIFF file.
    """
    with rasterio.open(tif_path) as src:
        # Read all bands (assume 3 bands: VV, VH, angle)
        bands = src.read()  # shape: (3, height, width)
        no_value = src.nodata
        print(f'NoData value from GeoTIFF data: {no_value}')
        # Flatten each band to 1D array
        bands_flat = bands.reshape(3, -1)  # shape: (3, num_pixels)
        # If no_data values is not None and not NaN, replace it with NaN for consistency
        if no_value is not np.nan:
            bands_flat = np.where(bands_flat == 0.0, np.nan, bands_flat)
        print(bands_flat.shape)  # Should print (3, num_pixels)
    return bands_flat

def get_coordinates(tif_path):
    """
    Return a numpy array of coordinates from a GeoTIFF file.
    """
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        # Generate row and column indices for all pixels
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        # Flatten arrays for vectorized operations
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        # Get (lon, lat) coordinates for each pixel's center 
        lon, lat = xy(src.transform, rows_flat, cols_flat, offset='center')
        # Stack the coordinates into a 2D array
        coordinates = np.column_stack((lon, lat))
    return coordinates

def extract_ndvi(ndvi_folder, coordinates):
    ndvi_image_files  = sorted([os.path.join(ndvi_folder, f) for f in os.listdir(ndvi_folder) if f.endswith('.tif')])
    dates = [extract_date(os.path.basename(f)) for f in ndvi_image_files]
    num_ndvi_images = len(ndvi_image_files) 

    # Initialize NDVI matrix 
    ndvi_matrix = np.full((len(coordinates), num_ndvi_images), np.nan, dtype=np.float32)
    print(f"NDVI matrix shape: {ndvi_matrix.shape}")
    for img_idx, tif_file in enumerate(ndvi_image_files):
        with rasterio.open(tif_file) as src:
            ndvi_values = list(src.sample(coordinates))
            ndvi_values = np.array(ndvi_values).squeeze()
            ndvi_matrix[:, img_idx] = ndvi_values

    # Prepare DataFrame with lat, lon, and NDVI values
    df_ndvi = pd.DataFrame(ndvi_matrix, columns=dates)
    df_ndvi.insert(0, "lon", [lon for lon, lat in coordinates])
    df_ndvi.insert(0, "lat", [lat for lon, lat in coordinates])

    # Save to CSV
    # df_nvdi.to_csv(f"{root_path}/{region}/csv_output/ndvi_values_tem.csv", index=False)
    # print("Saved NDVI csv!")
    return df_ndvi, dates, num_ndvi_images

def extract_era5(era5_folder, coordinates, dates, num_ndvi_images):
    era5_image_files = sorted([os.path.join(era5_folder, f'Weather_{date}.tif') for date in dates])
    T_matrix = np.full((len(coordinates), num_ndvi_images), np.nan, dtype=np.float32)
    P_matrix = np.full((len(coordinates), num_ndvi_images), np.nan, dtype=np.float32)
    for img_idx, tif_file in enumerate(era5_image_files):
        with rasterio.open(tif_file) as src:
            era5_values = list(src.sample(coordinates))
            T_values = np.array([val[0] for val in era5_values])
            P_values = np.array([val[1] for val in era5_values])
            T_matrix[:, img_idx] = T_values
            P_matrix[:, img_idx] = P_values

    # Prepare DataFrame with lat, lon, and NDVI values
    df_T = pd.DataFrame(T_matrix, columns=dates)
    df_T.insert(0, "lon", [lon for lon, lat in coordinates])
    df_T.insert(0, "lat", [lat for lon, lat in coordinates])

    # Save to CSV
    # df_T.to_csv(f"{root_path}/{region}/csv_output/T_values_tem.csv", index=False)
    # print("Saved Temperature csv!")

    df_P = pd.DataFrame(P_matrix, columns=dates)
    df_P.insert(0, "lon", [lon for lon, lat in coordinates])
    df_P.insert(0, "lat", [lat for lon, lat in coordinates])

    # Save to CSV
    # df_P.to_csv(f"{root_path}/{region}/csv_output/P_values_tem.csv", index=False)
    # print("Saved Precipitation csv!")
    return df_T, df_P

def get_mean_within_pixel_centers_multi_band(dem_src, polygon_geom):
    """
    Calculate mean value for each band of DEM, only for pixels whose center lies within polygon_geom.
    Handles polygons partially or fully outside DEM bounds
    
    Args:
        dem_src: rasterio opened dataset (multi-band)
        polygon_geom: shapely Polygon (cùng CRS với dem_src)

    Returns:
        np.array: Mảng trung bình cho từng band (shape: [num_bands])
    """
    # Get DEM bounds as a shapely box 
    dem_bounds = box(*dem_src.bounds)  # (minx, miny, maxx, maxy)
    # Intersect polygon with DEM bounds 
    intersection = polygon_geom.intersection(dem_bounds)
    if intersection.is_empty:
        # polygon is completely outside DEM bounds
        num_bands = dem_src.count 
        return np.array([np.nan] * num_bands, dtype=np.float32)
    # Use the intersection window for windowing and masking
    window = rasterio.features.geometry_window(dem_src, [intersection])
    data = dem_src.read(window=window, masked=True)  # shape: (bands, rows, cols)
    transform = dem_src.window_transform(window)

    num_bands, height, width = data.shape
    band_values = [[] for _ in range(num_bands)]

    for row in range(height):
        for col in range(width):
            # Tính tâm pixel
            x, y = rasterio.transform.xy(transform, row, col, offset='center')
            point = Point(x, y)
            if not polygon_geom.contains(point):
                continue

            for b in range(num_bands):
                val = data[b, row, col]
                if not np.ma.is_masked(val):
                    band_values[b].append(val)

    means = [np.nan if len(vals) == 0 else np.mean(vals) for vals in band_values]
    return np.array(means, dtype=np.float32)

def extract_terrain_soil_texture(soil_tif, dem_tif, coordinates, pobj = None, grid_size = 0.1, pg = None):
    # Extract soil values
    with rasterio.open(soil_tif) as src:
        soil_values = list(src.sample(coordinates))
        soil_values = np.array(soil_values)  # shape: (N, 3)
        # If shape is (N,), reshape to (N, 1)
        if soil_values.ndim == 1:
            soil_values = soil_values[:, np.newaxis]

    # Extract DEM values
    with rasterio.open(dem_tif) as dem_src:
        dem_means = []
        count = 0

        for coord in coordinates:
            lon, lat = coord
            ring_wgs, grid_ring = pobj.get_wgs_grid(lon, lat)
            grid_geom_shapely = Polygon(ring_wgs)  # EPSG:4326
            transformer = Transformer.from_crs("EPSG:4326", dem_src.crs, always_xy=True)
            grid_geom_proj = Polygon([transformer.transform(x, y) for x, y in grid_geom_shapely.exterior.coords])

            mean_elev = get_mean_within_pixel_centers_multi_band(dem_src, grid_geom_proj)
            print(f"{count} :{mean_elev}")
            count += 1
            dem_means.append(mean_elev)


    dem_values = np.array(dem_means)

    # Tách các thành phần
    sand, clay, bdod = soil_values[:, 0], soil_values[:, 1], soil_values[:, 2]
    elevation, slope, aspect_deg = dem_values[:, 0], dem_values[:, 1], dem_values[:, 2]

    aspect_rad = np.deg2rad(aspect_deg)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]

    df = pd.DataFrame({
        'sand_0-5cm_mean': sand,
        'clay_0-5cm_mean': clay,
        'bdod_0-5cm_mean': bdod,
        'elevation': elevation,
        'slope': slope,
        'aspect_sin': aspect_sin,
        'aspect_cos': aspect_cos,
    })

    reprojected_coords = [pg.re_project(lon, lat) for lon, lat in zip(lons, lats)] if pg else coordinates
    x_proj = [coord[0] for coord in reprojected_coords]
    y_proj = [coord[1] for coord in reprojected_coords]

    # Sau đó đưa vào pobj.xy_2_cr
    r, c = pobj.xy_2_cr(x_proj, y_proj)

    # Gán r, c từ pobj
    df['r'] = np.array(r) / (9 / grid_size) / 1624
    norm_c = np.array(c) / (9 / grid_size) / 3856
    df['c_sin'] = np.sin(norm_c * 2 * np.pi)
    df['c_cos'] = np.cos(norm_c * 2 * np.pi)

    df['lon'] = lons
    df['lat'] = lats
    return df 

def download_image(image, geometry, folder, image_id, resolution, crs = 'EPSG:4326'):
    local_path = os.path.join(folder, f"{image_id}.tif")
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping")
        return
    url = image.getDownloadURL({
        'scale': resolution,
        'crs': crs,
        'region': geometry,
        'filePerBand': False, 
        'format': 'GeoTIFF',
    })
    
    local_path = os.path.join(folder, f"{image_id}.tif")
    print(f"Downloading {image_id} to {local_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad requests 
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {image_id} successfully.")
    except Exception as e:
        print(f"Failed to download {image_id}: {e}")
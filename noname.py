import rasterio 
tif_path = '/mnt/data2tb/nsidc_images/NSIDC-0779_EASE2_G1km_SMAP_SM_DS_20220117.tif'
with rasterio.open(tif_path) as src:
        print("Raster CRS:", src.crs)
        raster_crs = src.crs
        print("RASTER CRS:",raster_crs)
import numpy as np
import cv2 
import pandas as pd 
import rasterio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
args = parser.parse_args()

region = args.region
# Create necessary a directory to store output images
os.makedirs(f'roi_inference/regions_data_results/{region}/sm', exist_ok=True)

# def scale_to_uint8(arr, arr_min = 0.1, arr_max = 0.7):
#     arr_scaled = (arr - arr_min) / (arr_max - arr_min)
#     arr_scaled = np.clip(arr_scaled, 0, 1)
#     arr_uint8 = (arr_scaled * 255)
#     arr_uint8[np.isnan(arr)] = 0  # Set NaN to 0
#     return arr_uint8.astype(np.uint8)

# # Save as green channel PNGs
# def save_green_png(arr_uint8, path, scale, mask = None, is_ground_truth = False):
#     h, w = arr_uint8.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     rgb[..., 1] = arr_uint8
#     # Resize the image to 8x larger using nearest neighbor for clear pixel boundaries
#     rgb_large = cv2.resize(rgb, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
#     if mask is not None:
#         if mask.shape != rgb_large.shape[:2]:
#             raise ValueError("Mask shape does not match resized image shape")
#         # Apply mask: set all non-agriculture (mask==0) pixels to black
#         rgb_large[mask == 0] = 0

#     if is_ground_truth:
#         gray = rgb_large[:,:,1]
#         zero_mask = (gray == 0).astype(np.uint8)*255

#         # Find contours surrouding area that has no data 
#         contours, _ = cv2.findContours(zero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(rgb_large, contours, -1, (0, 0, 255), 3)
#     cv2.imwrite(path, rgb_large)

target_dates = []
combination_folder = f'roi_inference/regions_data_results/{region}/csv_output/combination'

# Get all dates that was predicted 
target_dates = []
for filename in os.listdir(combination_folder):
    if filename.endswith('.csv'):
        date = filename.split('.')[0].split('_')[0]
        target_dates.append(date)

# Read a Sentinel-1 iamge to get the metadata for saving GeoTIFF
with rasterio.open(f'roi_inference/regions_data_results/{region}/data/s1_images/S1_{target_dates[0]}.tif') as src:
    profile = src.profile 
    transform = src.transform 
    crs = src.crs 
    image_shape = src.shape

# Update the profile because we just need one band for the soil moisture
profile.update(
    {
        'count' : 1,
        'dtype' : 'float32',
    }
)

# Traverse through the file of each date and save the predictions as GeoTIFF 
for target_date in target_dates:
    print(target_date)
    predicted = pd.read_csv(f'roi_inference/regions_data_results/{region}/prediction/{target_date}.csv')
    print('predicted: ',len(predicted))
    predicted_values = np.array(predicted['Prediction']).reshape(image_shape)
    pred = predicted_values[(predicted_values > 0) & (predicted_values < 0.7)]
    print(f"Predicted values mean ({target_date}): {np.nanmean(pred)}")
    output_path = os.path.join(f'roi_inference/regions_data_results/{region}/sm',f'{target_date}.tif')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predicted_values.astype('float32'), 1)
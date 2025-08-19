import numpy as np
import pandas as pd 
import rasterio
import os
import argparse
import shutil
import datetime as dt
from bisect import bisect_left
from pathlib import Path

def create_8_days_sm_images(src_folder, dst_folder):
    """
    Tạo ảnh cách nhau 8 ngày, neo theo 01-01 mỗi năm nhưng CHỈ trong
    khoảng [min_date, max_date] của dữ liệu gốc. Nếu thiếu ngày đúng
    thì lấy ngày gần nhất (nếu khoảng cách bằng nhau -> ưu tiên ngày SAU).
    """
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    # Đọc tên file dạng YYYY-MM-DD.tif -> map {date: filepath}
    files = {}
    for f in src.glob("*.tif"):
        try:
            d = dt.date.fromisoformat(f.stem)
            files[d] = f
        except ValueError:
            print(f"Ignore the non valid file: {f.name}")

    if not files:
        print(f"Thera are no .tif files in {src}")
        return

    available = sorted(files.keys())
    min_d, max_d = available[0], available[-1]

    # Sinh các mốc 8 ngày (neo 01-01 mỗi năm) rồi CHỈ giữ mốc nằm trong [min_d, max_d]
    targets = []
    for y in range(min_d.year, max_d.year + 1):
        start = dt.date(y, 1, 1)
        for i in range(46):
            t = start + dt.timedelta(days=8 * i)
            if min_d <= t <= max_d:
                targets.append(t)

    # Tìm ngày gần nhất bằng bisect (nếu hòa -> ưu tiên ngày SAU)
    def closest_date(x):
        i = bisect_left(available, x)
        left = available[i-1] if i > 0 else None
        right = available[i] if i < len(available) else None
        if left and right:
            dl = (x - left).days
            dr = (right - x).days
            if dr < dl:
                return right
            elif dl < dr:
                return left
            else:
                return right  # hòa thì chọn ngày sau (>= target)
        return right or left

    # Copy và đặt tên theo ngày target
    for t in targets:
        c = closest_date(t)
        shutil.copy(files[c], dst / f"{t.isoformat()}.tif")
        print(f"{t} <- copy from {c}")

    print(f"Save {len(targets)} image in {dst}")

parser = argparse.ArgumentParser()
parser.add_argument('--region', required=True)
parser.add_argument('--data_folder', required = True)
args = parser.parse_args()

region = args.region
data_folder = args.data_folder
# Create necessary a directory to store output images
os.makedirs(f'{data_folder}/{region}/sm', exist_ok=True)

output_sm_folder = f'{data_folder}/{region}/sm'
sm_folder_8days = f'{data_folder}/{region}/sm_8days'

target_dates = []
combination_folder = f'{data_folder}/{region}/csv_output/combination'

# Get all dates that was predicted 
target_dates = []
for filename in os.listdir(combination_folder):
    if filename.endswith('.csv'):
        date = filename.split('.')[0].split('_')[0]
        target_dates.append(date)

# Read a Sentinel-1 iamge to get the metadata for saving GeoTIFF
with rasterio.open(f'{data_folder}/{region}/data/s1_images/S1_{target_dates[0]}.tif') as src:
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
    if not os.path.exists(f'{data_folder}/{region}/prediction/{target_date}.csv'):
        print(f"Data in {target_date} is missed so can not run inference!!!")
        continue
    predicted = pd.read_csv(f'{data_folder}/{region}/prediction/{target_date}.csv')
    print('predicted: ',len(predicted))
    predicted_values = np.array(predicted['Prediction']).reshape(image_shape)
    pred = predicted_values[(predicted_values > 0) & (predicted_values < 0.7)]
    print(f"Predicted values mean ({target_date}): {np.nanmean(pred)}")
    output_path = os.path.join(output_sm_folder,f'{target_date}.tif')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predicted_values.astype('float32'), 1)

# Create folder containg sm data for each 8 days
create_8_days_sm_images(output_sm_folder, sm_folder_8days)

import numpy as np
import pandas as pd 
import rasterio
import os
import argparse
import shutil
import datetime
from pathlib import Path

def create_8_days_sm_images(src_folder, dst_folder):
    """
    Lọc và copy ảnh TIFF cách nhau 8 ngày, bắt đầu từ YYYY-01-01 cho từng năm.
    Nếu không có ngày đúng thì lấy ngày gần nhất.
    Luôn tạo đủ 46 ngày trong năm.

    Args:
        src_folder (str): thư mục gốc chứa TIFF
        dst_folder (str): thư mục kết quả
    """
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách file gốc và chuyển thành dict {date: filepath}
    files = {}
    for f in src_folder.glob("*.tif"):
        try:
            d = datetime.date.fromisoformat(f.stem)  # stem = tên file không có .tif
            files[d] = f
        except Exception:
            print(f"Ignore if the file is not valid: {f.name}")

    if not files:
        print(f"There are no tif image in the {src_folder}.")
        return

    available_dates = sorted(files.keys())

    # Nhóm theo năm
    years = sorted(set(d.year for d in available_dates))

    for year in years:
        print(f"Processing {year}...")
        # target dates: 46 ngày cách nhau 8 ngày
        start_date = datetime.date(year, 1, 1)
        target_dates = [start_date + datetime.timedelta(days=8 * i) for i in range(46)]

        for target in target_dates:
            # tìm ngày gần nhất trong tất cả available_dates
            closest = min(available_dates, key=lambda d: abs(d - target))
            src_file = files[closest]

            dst_file = dst_folder / f"{target.isoformat()}.tif"
            shutil.copy(src_file, dst_file)
            print(f"{target} <- copy from {closest}")

    print(f"Saved output images in {dst_folder}")

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

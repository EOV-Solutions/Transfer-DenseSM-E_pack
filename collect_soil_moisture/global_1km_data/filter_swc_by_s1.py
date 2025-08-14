""" This script filters 1km soil moisture data based on Sentinel-1 dates for a specific region.
    We just keep the soil moisture data that has a date in the Sentinel-1 metadata or 1-day difference.
    
    tree_grass_crops_csv ===> tree_grass_crops_csv_filtered"""

import pandas as pd 
import os 
import glob

REGION = 'india' # or 'china'
network = 'INDIA_1km' # or 'CHINA_1km'
root_path = "/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/1km_global"
csv_folder = f'{root_path}/{REGION}/tree_grass_crops_csv'
output_folder = f'{root_path}/{REGION}/tree_grass_crops_csv_filtered'
s1_date_path = f'{root_path}/{REGION}/{REGION}_s1_metadata.csv'

os.makedirs(output_folder, exist_ok=True)
def filter_swc():
    s1_date_df = pd.read_csv(s1_date_path)
    s1_date_df['date'] = pd.to_datetime(s1_date_df['date'])
    # Get days from Sentinel-1 metadata 
    s1_dates = set(s1_date_df['date'])

    # Loop through each CSV file of points in the folder to filter
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_folder, filename)
            swc_df = pd.read_csv(file_path)
            swc_df = swc_df.drop(columns=['date'])
            id = filename.split('.')[0]
            
            swc_df['time'] = pd.to_datetime(swc_df['time'])

            # Tạo bản sao
            adjusted_swc_df = swc_df.copy()

            # Tính ngày time - 1
            adjusted_swc_df['time_minus1'] = adjusted_swc_df['time'] - pd.Timedelta(days=1)

            # Tạo điều kiện ưu tiên time là ngày có dữ liệu soil moisture:
            # 1. Nếu time thuộc s1_dates → giữ nguyên
            # 2. Nếu time - 1 thuộc s1_dates → gán time = time - 1
            # 3. Nếu không thuộc 2 trường hợp trên → loại

            # Tạo mask cho từng trường hợp
            mask_time_in_s1 = adjusted_swc_df['time'].isin(s1_dates)
            mask_time_minus1_in_s1 = adjusted_swc_df['time_minus1'].isin(s1_dates)

            # Gán time = time - 1 nếu time không thuộc nhưng time -1 thuộc
            adjusted_swc_df.loc[~mask_time_in_s1 & mask_time_minus1_in_s1, 'time'] = adjusted_swc_df['time_minus1']

            # Giữ lại các dòng có time cuối cùng nằm trong s1_dates
            filtered_swc_df = adjusted_swc_df[adjusted_swc_df['time'].isin(s1_dates)].drop(columns=['time_minus1'])
            
            output_path = os.path.join(output_folder, filename)

            filtered_swc_df.to_csv(output_path, index = False)


# def merge_filtered_swc():
#     df_list = []
#     filtered_csv_path = glob.glob(os.path.join(output_folder, '*.csv'))
#     for path in filtered_csv_path:
#         filtered_swc = pd.read_csv(path)
#         filtered_swc = filtered_swc.dropna()
#         df_list.append(filtered_swc)

#     merged_filtered_swc = pd.concat(df_list, ignore_index= True)

#     merged_filtered_swc = merged_filtered_swc.drop_duplicates(subset=['sm'])
#     merged_filtered_swc.to_csv(f'{root_path}/{REGION}/points/merged.csv')

def create_site_info_file():
    s_depth = 0
    e_depth = 5
    station_list = []
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        df = pd.read_csv(file_path)

        if df.empty:
            continue

        id = int(filename.split('.')[0])
        lat = df.iloc[0]['latitude']
        lon = df.iloc[0]['longitude']
        station = {
            'network' : network, 
            'station' : id,
            'lat'     : lat, 
            'lon'     : lon, 
            's_depth' : s_depth,
            'e_depth' : e_depth,
        }

        station_list.append(station)

    # Convert to DataFrame 
    station_df = pd.DataFrame(station_list)
    station_df.to_csv(f'{root_path}/{REGION}/points/tree_grass_crops_site_info.csv', index=False)


def main():
    filter_swc()
    # After filtering, if there are points with no soil moisture data, we should create a site info file again
    # create_site_info_file()

if __name__ == "__main__":
    main()
    
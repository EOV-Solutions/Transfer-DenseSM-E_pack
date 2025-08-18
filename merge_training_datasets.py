import pandas as pd
import numpy as np

# Fucntion to balance the data based on soil moisture (sm) values
def balance_data(data):
    df = data[(data['sm'] >= 0.1) & (data['sm'] <= 0.7)]
    # Balance the data based on soil moisture (sm) values, we split the data into three categories: <= 0.3, 0.3 < sm < 0.4, and >= 0.4
    central_mask = (df['sm'] >= 0.3) & (df['sm'] <= 0.4)
    low_mask = df['sm'] < 0.3 
    high_mask = df['sm'] > 0.4

    # Keep 50% of the low (<= 0.3) and high (>=0.4) soil moisture samples, and all central samples
    df_central = df[central_mask]
    df_low = df[low_mask].sample(frac = 0.5, random_state = 42)
    df_hight = df[high_mask].sample(frac = 0.5, random_state = 42)

    # After balancing, we concatenate the three dataframes
    df_balanced = pd.concat([df_central, df_low, df_hight], axis = 0).reset_index(drop = True)
    return df_balanced
    
# Define path to CSV files
path1 = '100m_data/india/ndvi/india_landsat_merged.csv' # India data 100m
path2 = '100m_data/china/ndvi/china_landsat_merged.csv' # China data 100m
path3 = '1km_data/vn_points/ndvi/vn_landsat_merged.csv' # Vietnam data 1km
path4 = '1km_global_data/india/ndvi/india1km_landsat_merged.csv' # India data 1km
path5 = '1km_global_data/china/ndvi/china1km_landsat_merged.csv' # China data 1km

data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
data3 = pd.read_csv(path3)
data4 = pd.read_csv(path4)
data5 = pd.read_csv(path5)                                                 

print(f"The number of rows in india 100m: {len(data1)}") 
print(f"The number of rows in china 100m: {len(data2)}")                                
print(f"The number of rows in vietnam 1km: {len(data3)}")                                
print(f"The number of rows in india 1km: {len(data4)}")                                
print(f"The number of rows in china 1km: {len(data5)}") 

data_100 = pd.concat([data1, data2], axis = 0)
# Do not need to add location noise to the 100m data
# for col in ['12', '13', '14']:
#     noise = np.random.normal(loc = 0.0, scale = 0.1, size = len(data_100))
#     data_100[col] = np.clip(data_100[col] + noise, 0, 1)

data = pd.concat([data_100, data3, data4, data5], axis = 0)
data['s_index'] = range(1, len(data) + 1)
data.to_csv('/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/fusion/file_name.csv', index = False)

# Balance the data based on soil moisture (sm) values for better training
balanced_data = balance_data(data)
# Save the balanced dataframe to a CSV file
balanced_data.to_csv('/mnt/data2tb/Transfer-DenseSM-E_pack/training_data/fusion/balanced_deletethis.csv', index = False)
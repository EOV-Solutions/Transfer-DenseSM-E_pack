"""
Mục đích của chương trình này là xét các vùng có dữ liệu soil moisture, xác định các ngày có dữ liệu Sentinel-1 tại các vùng đó. 
Từ đó dùng cho các chương trình sau này để lọc giá trị sm, chỉ giữ lấy các giá trị sm có cùng ngày với S1 hoặc sau S1 1 ngày.
File metedata thời gian này có thể dùng chung khi lọc dữ liệu 100m và 1km tại Ấn Độ và Trung Quốc, 
vì cùng lấy trên một khu vưc giống nhau (1km thì sẽ lấy trên vùng bao quanh vùng có dữ liệu 100m)
"""

import ee 
import pandas 

ee.Initialize()

# Define the region of interest as a polygon
polygon_coords = [[
        [102.1445, 23.3937],
        [109.4462, 23.3937],
        [109.4462, 8.3439],
        [102.1445, 8.3439],
        [102.1445, 23.3937]
    ]]

"""{"type":"Polygon","coordinates":[[
[37.95826023542673,-1.230450106759613],
[37.95762785043055,-1.447358160448246],
[37.741352181738094,-1.447990545444423],
[37.74261695173045,-1.229817721763436],
[37.95826023542673,-1.230450106759613]]]}"""

region = ee.Geometry.Polygon(polygon_coords)

start_date = '2017-01-01'
end_date = '2022-12-31'

# Create a Sentinel-1 image collection get both ASC and DESC orbit passes
collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(region) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .filter(ee.Filter.eq('platform_number', 'A')) \
    .select(['VV', 'VH']) 

# Extract metadata (date + orbit pass) from the images
def extract_metadata(image):
    return ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd'),
        'orbit_pass': image.get('orbitProperties_pass')
    })

features = collection.map(extract_metadata).distinct(['date', 'orbit_pass'])
date = features.aggregate_array('date').getInfo()
orbit_pass = features.aggregate_array('orbit_pass').getInfo()

# Create a DataFrame from the metadata 
df = pandas.DataFrame({
    'date': date,
    'orbit_pass': orbit_pass
})

df['time'] = df['date'].astype(str)
df = df.sort_values(by='date', ascending=True)

# Replace 'your_path.csv' with the desired output path
df.to_csv('your_path.csv', index=False)
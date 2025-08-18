"""
Mục đích của chương trình này là xét các vùng có dữ liệu soil moisture, xác định các ngày có dữ liệu Sentinel-1 tại các vùng đó. 
Từ đó dùng cho các chương trình sau này để lọc giá trị sm, chỉ giữ lấy các giá trị sm có cùng ngày với S1 hoặc sau S1 1 ngày.
File metedata thời gian này có thể dùng chung khi lọc dữ liệu 100m và 1km tại Ấn Độ và Trung Quốc, 
vì cùng lấy trên một khu vưc giống nhau (1km thì sẽ lấy trên vùng bao quanh vùng có dữ liệu 100m)
"""

import ee 
import pandas 

ee.Initialize()

def get_s1_dates(polygon_coords, start_date, end_date, output_path):
    # Define the region of interest as a polygon
    region = ee.Geometry.Polygon(polygon_coords)

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

    # Save to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Saved the Sentinel-1 date to {output_path}")

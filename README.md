# Transfer-DenseSM-E
# Soil Moisture Estimation Project

## Giới Thiệu 
Dự án này xây dựng pipeline xử lý, huấn luyện và suy luận (inference) dữ liệu **độ ẩm đất (Soil Moisture - SM) độ phân giải 100m**. Sử dụng ground truth từ nhiều nguồn Planet Variable (100m) và NSDIC (1km). Đầu vào là các nguồn dữ liệu viễn thám bao gồm: Sentinel-1, MODIS-NDVI, Temperatur, Precipiation, SoilGrids, DEM.

Quy trình bao gồm:
- Tải dữ liệu soil moisture từ các nguồn Planet(Hiện tại không cho download từ Sentinelhub EO Browser) và NSIDC (https://nsidc.org/data/spl4smgp/versions/7)
- Áp dụng grid và dữ liệu landcover để lựa chọn các điểm lấy dữ liệu. 
- Trích xuất và xử lý dữ liệu soil moisture ở các độ phân giải (100m, 1km)
- Tải các dữ liệu đầu vào khác (Sentinel-1, NDVI, Soil Texture,...), kết hợp với dữ liệu soil moisture.
- Huấn luyện mô hình **DensSM**.
- Thực hiện inference cho các vùng quan tâm và trực quan hóa kết quả. 

## ⚙️ Chuẩn bị môi trường
Chạy script để tạo môi trường 'conda':
```bash
bash setup_env.sh
conda activate sm
```
---

## 📂 Cấu trúc dữ liệu
- **training_data/**
  - `100m/` : dữ liệu độ ẩm đất 100m  
  - `1km_vn/` : dữ liệu 1km tại Việt Nam  
  - `1km_global/` : dữ liệu 1km toàn cầu  
  - `fusion/` : dữ liệu đã ghép phục vụ huấn luyện  
- **pretrained_models/** : chứa các mô hình DenseSM có sẵn  
- **trained_models/** : lưu các mô hình sau khi fine-tune  
- **roi_inference/** : code và dữ liệu cho inference theo vùng 

## 🔮 Inference cho vùng quan tâm
File chính: roi_inference/run_pipeline.py
```bash
python roi_inference/run_pipeline.py     --region <region_name>     --start_date 2024-12-25     --end_date 2025-02-05     --download --extract --process --inference --visualize
```
Các bước trong pipeline:
- `download` : tải dữ liệu  
- `extract` : trích xuất thành CSV  
- `process` : tiền xử lý, ghép NDVI, DEM, v.v.  
- `inference` : chạy DenseSM ensemble  
- `visualize` : xuất kết quả thành ảnh `.tif`  

Có thể chạy riêng từng bước, ví dụ khi đã chạy `download` và `extract` thì chỉ cần chạy 3 bước còn lại:
```bash
python roi_inference/run_pipeline.py     --region ngocnhat2     --start_date 2024-12-25     --end_date 2025-02-05     --process --inference --visualize
```

## 📥 Tải và xử lý dữ liệu Soil Moisture
### Quy trình chung:
1. Chia vùng thành **grid** (1km, 2km, …).  
2. Kết hợp với **land cover** để lọc và chọn điểm.  
3. Trích xuất **soil moisture** từ ảnh `.tif` (Planet / NSIDC).  
4. Lọc theo ngày có Sentinel-1.  
5. Tạo file `.csv` chứa thông tin phục vụ huấn luyện. 


## Part II: multi-scale domain adpation method (MSDA)

The pretrained 9km models were in DenseSM_9km.zip, while the samples for CONUS is in samples.zip https://doi.org/10.5281/zenodo.13336185

The initial version of finetune (Zhu et al., 2024) was inlcuded in MSDA.

Use example.ipynb to run the MSDA.


## Reference
Liujun Zhu, Junjie Dai, Yi Liu, Shanshui Yuan & Jeffrey P. Walker (2024) A cross-resolution transfer learning approach for soil moisture retrieval with limited training samples, Remote Sensing of Environment
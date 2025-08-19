# Transfer-DenseSM-E -- Soil Moisture Estimation Project

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

`collect_100m_sm.py`: Xử lý sm 100m cho 'china' hoặc 'india'
`collect_1km_global_sm.py`: Xử lý sm 1km cho 'china' hoặc 'india'
`collect_1km_vn_sm.py`: Xử lý sm 1km cho Việt Nam

Kết quả:
- **Site information file**: chứa thông tin tất cả điểm trong vùng  
- **CSV folder**: mỗi file chứa thông tin độ ẩm cho từng điểm  

---

## 📊 Chuẩn bị dữ liệu huấn luyện
Chạy file `data_pre/prepare_samples.py` để:
- Đọc danh sách site và SM values.  
- Ghép thêm dữ liệu đầu vào NDVI, LST, DEM, Precipitation... cùng với các giá trị sm tương ứng 
- Xuất file dữ liệu trainng cho model.  

Chạy file `merge_training_datasets.py` để kết hợp các loại dữ liệu với nhau thành 1 file tổng hợp (fusion) lưu trong thư mục **training_data\fusion**.

---

## 🧠 Huấn luyện mô hình
Mô hình **DenseSM** được dùng để huấn luyện với dữ liệu fusion.  

Yêu cầu đầu vào:
- File `.csv` trong `training_data/fusion`, tùy vào cách kết hợp các loại dữ liệu sẽ cho ra các file tương ứng.   
- Pretrained models trong `pretrained_models/DenseSM_9km`  

Huấn luyện nhiều vòng lặp:
```python
for r in range(3):
    # train 25 models 3 lần
```
Kết quả lưu trong `trained_models/ft12_model/`.

---


## Reference
Liujun Zhu, Junjie Dai, Yi Liu, Shanshui Yuan & Jeffrey P. Walker (2024) A cross-resolution transfer learning approach for soil moisture retrieval with limited training samples, Remote Sensing of Environment
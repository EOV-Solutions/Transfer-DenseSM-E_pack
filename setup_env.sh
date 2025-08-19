#!/bin/bash
# Tạo môi trường Python 3.11
conda create -y -n sm python=3.11

# Kích hoạt môi trường
# ⚠️ Lưu ý: trong script bash, "conda activate" cần khởi tạo trước
# nên có thể chạy thủ công sau khi script xong.
conda activate sm

# Cài các thư viện qua conda-forge
conda install -y -c conda-forge fiona pyogrio geopandas gdal shapely rasterio
conda install -y -c conda-forge opencv
conda install -y -c conda-forge "numpy<2"

# Cài các thư viện pip
pip install earthengine-api
pip install geemap

# Cài PyTorch CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

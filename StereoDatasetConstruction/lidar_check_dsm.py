'''
Project    : RSDeploy
FileName   : lidar_check_dsm .py
CreateTime : 2025/4/19 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import csv

# 文件夹路径
folder_path = '/home/dshare/06Test/300ZDQY/1/Fuzhu/20250409_ICESAT2/GF07_BPA_026545_20240621_MH4A1_01_101_L1A_01_boundary'  # 替换为你的文件夹路径
excel_extension = '.csv'  # 或者 '.xlsx'，根据实际文件扩展名调整

# 存储所有点的列表
all_lons = []
all_lats = []
all_z_values = []

all_use = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(excel_extension):
        excel_file = os.path.join(folder_path, file_name)
        try:
            # 读取Excel文件
            df = pd.read_csv(excel_file)

            # 检查所需的列是否存在
            required_columns = {'lon_ph', 'lat_ph', 'h_ph'}
            if not required_columns.issubset(df.columns):
                print(f"文件 {file_name} 缺少必要的列: {required_columns - set(df.columns)}")
                continue

            # 提取经度、纬度和高程
            lons = df['lon_ph'].values
            lats = df['lat_ph'].values
            z_values = df['h_ph'].values

            # 添加到总列表中
            all_lons.extend(lons)
            all_lats.extend(lats)
            all_z_values.extend(z_values)

        except Exception as e:
            print(f"读取文件 {file_name} 时出错: {e}")

# 存储数据值
lid_values = []
dsm_values = []
dsm_values_DL = []

# 检查是否有有效数据
if not all_lons or not all_lats or not all_z_values:
    print("没有找到有效的点数据。")
else:
    # 读取遥感影像
    image_path = '/home/dshare/06Test/300ZDQY/1/TD/GF07_DLC_026545_20240621_MH4A1_01_101_L1A_01_202503240000000042.tif'
    image_path_DL = '/home/dshare/06Test/300ZDQY/1/DL/GF07_BPA_026545_20240621_MH4A1_01_101_L1A_01.tif'
    # image_path = '/home/dshare/06Test/300ZDQY/1/Fusion/Fusion2/GF07_DLC_026545_20240621_MH4A1_01_101_L1A_01_202503240000000042.tif'
    with rasterio.open(image_path) as src:
        img_crs = src.crs
        geotransform = src.transform
        img_width = src.width
        img_height = src.height
        img_array = src.read([1])  # 读取RGB波段

    with rasterio.open(image_path_DL) as src_DL:
        img_crs_DL = src_DL.crs
        geotransform_DL = src_DL.transform
        img_width_DL = src_DL.width
        img_height_DL = src_DL.height
        img_array_DL = src_DL.read([1])  # 读取RGB波段

    for idx, (lon, lat, lidar_z) in enumerate(zip(all_lons, all_lats, all_z_values)):
        try:
            # 获取对应站点的行列号
            row, col = src.index(lon, lat)
            row_DL, col_DL = src_DL.index(lon, lat)

            # 范围检查
            if not ((0 <= col < img_width-1) and (0 <= row <= img_height-1)):
                all_use.append(False)
                continue
            if not ((0 <= col_DL < img_width_DL-1) and (0 <= row_DL <= img_height_DL-1)):
                all_use.append(False)
                continue

            dsm_value = img_array[0][row, col]
            dsm_value_DL = img_array_DL[0][row_DL, col_DL]

            # dsm检查
            if dsm_value <= -999:
                all_use.append(False)
                continue
            if dsm_value_DL <= -999:
                all_use.append(False)
                continue

            lid_values.append(lidar_z)
            dsm_values.append(dsm_value)
            dsm_values_DL.append(dsm_value_DL)

            all_use.append(True)

            print("TD:" + str(dsm_value-lidar_z) + "  DL:" + str(dsm_value_DL-lidar_z))

        except Exception as e:
            print(f"文件 {file_name} 的第 {idx + 1} 行坐标转换或像素提取时出错: {e}")


# 计算差值
disp = np.array(dsm_values) - np.array(lid_values)
disp_DL = np.array(dsm_values_DL) - np.array(lid_values)

disp_mask = ((np.abs(disp) < 10) & (np.abs(disp_DL) < 10))
# disp_mask = np.abs(disp) < 10
disp_ = disp[disp_mask]
# all_use[all_use][disp < 10] = False
mean_value = np.mean(np.abs(disp_))
rmse_value = np.sqrt(np.mean(disp_**2))

# disp_mask_DL = np.abs(disp_DL) < 10
disp_DL_ = disp_DL[disp_mask]
mean_value_DL = np.mean(np.abs(disp_DL_))
rmse_value_DL = np.sqrt(np.mean(disp_DL_**2))

print(f"有效点数量是: {len(disp_)}")
print(f"TD平均误差值是: {mean_value}")
print(f"TD均方根误差是: {rmse_value}")

print(f"DL平均误差值是: {mean_value_DL}")
print(f"DL均方根误差是: {rmse_value_DL}")


if __name__ == '__main__':
    pass

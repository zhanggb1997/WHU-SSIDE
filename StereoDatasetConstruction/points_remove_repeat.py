'''
Project    : RSDeploy
FileName   : points_2_dsm .py
CreateTime : 2025/5/7 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import csv
import gc
import math
import os.path
import pylas
import cv2
import laspy
import numpy as np
import pyproj
import rasterio
from laspy.lib import create_las
from pyproj import CRS, Transformer
from rasterio import float32
from rasterio._warp import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from points_process import points_Z_IQR_filter, las_voxel_max_sampling, SelfLas, laspy_voxel_max_sampling


def point_weiyi(las_paths, output_path):
    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    chunk_size = 5000  # 500米
    precision = 0.01  # 坐标精度容差(米)，默认0.01(厘米级)

    # ============= 读取las =============
    # with laspy.open(las_paths[0]) as reader0, laspy.open(las_paths[1]) as reader1, laspy.open(las_paths[2]) as reader2, laspy.open(las_paths[3]) as reader3, laspy.open(las_paths[4]) as reader4, laspy.open(las_paths[5]) as reader5, laspy.open(las_paths[6]) as reader6:
    with laspy.open(las_paths[0]) as reader0, laspy.open(las_paths[1]) as reader1, laspy.open(las_paths[2]) as reader2:
        # 合并多个文件的范围
        min_x = min(reader0.header.mins[0], reader1.header.mins[0], reader2.header.mins[0])
        max_x = max(reader0.header.maxs[0], reader1.header.maxs[0], reader2.header.maxs[0])
        min_y = min(reader0.header.mins[1], reader1.header.mins[1], reader2.header.mins[1])
        max_y = max(reader0.header.maxs[1], reader1.header.maxs[1], reader2.header.maxs[1])

        #
        # min_x = min(reader0.header.mins[0], reader1.header.mins[0], reader2.header.mins[0], reader3.header.mins[0], reader4.header.mins[0], reader5.header.mins[0], reader6.header.mins[0])
        # max_x = max(reader0.header.maxs[0], reader1.header.maxs[0], reader2.header.maxs[0], reader3.header.maxs[0], reader4.header.maxs[0], reader5.header.maxs[0], reader6.header.maxs[0])
        # min_y = min(reader0.header.mins[1], reader1.header.mins[1], reader2.header.mins[1], reader3.header.mins[1], reader4.header.mins[1], reader5.header.mins[1], reader6.header.mins[1])
        # max_y = max(reader0.header.maxs[1], reader1.header.maxs[1], reader2.header.maxs[1], reader3.header.maxs[1], reader4.header.maxs[1], reader5.header.maxs[1], reader6.header.maxs[1])
        #
        # 确保坐标范围有适当的缓冲区
        buffer = chunk_size * 0.1
        min_x -= buffer
        max_x += buffer
        min_y -= buffer
        max_y += buffer

        # 计算分块网格
        x_blocks = int(np.ceil((max_x - min_x) / chunk_size))
        y_blocks = int(np.ceil((max_y - min_y) / chunk_size))

        # 获取点格式并准备输出
        header = reader0.header
        # pt_format = reader0.point_format
        # points_to_write = []
        seen_points = set()

        # 步骤2：逐块处理点云
        with laspy.open(output_path, mode="w", header=header) as writer:
            with tqdm(total=x_blocks * y_blocks, desc="处理空间分块") as pbar:
                for x_idx in range(x_blocks):
                    for y_idx in range(y_blocks):
                        # 计算当前块范围
                        block_min_x = min_x + x_idx * chunk_size
                        block_max_x = min_x + (x_idx + 1) * chunk_size
                        block_min_y = min_y + y_idx * chunk_size
                        block_max_y = min_y + (y_idx + 1) * chunk_size

                        block_points = []

                        # 从文件1读取当前块的点
                        for reader in [reader0, reader1, reader2]:
                        # for reader in [reader0, reader1, reader2, reader3, reader4, reader5, reader6]:
                            points_in_block = []
                            for points in reader.chunk_iterator(100_000):
                                mask = (points.x >= block_min_x) & (points.x <= block_max_x) & \
                                       (points.y >= block_min_y) & (points.y <= block_max_y)

                                if np.any(mask):
                                    points_in_block.extend(points[mask])

                            # 添加当前文件中的点
                            for point in points_in_block:
                                # 量化坐标到指定精度
                                x_key = round(point.x / precision) * precision
                                y_key = round(point.y / precision) * precision
                                z_key = round(point.z / precision) * precision
                                coord_key = (x_key, y_key, z_key)

                                # 检查是否已存在相同坐标点
                                if coord_key not in seen_points:
                                    seen_points.add(coord_key)
                                    block_points.append(point)

                        # 写入当前块的唯一坐标点
                        if block_points:
                            writer.write_points(block_points)
                            points_to_write = []  # 重置点缓存

                        pbar.update(1)

    print(f"处理完成！输出文件: {output_path}")




def point_merge(las_paths, output_path):
    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    # ============= 读取las =============
    with laspy.open(las_paths[0]) as reader0:

        # 获取点格式并准备输出
        header = reader0.header
        pt_format = reader0.header.point_format

        # 步骤2：逐块处理点云
        with laspy.open(output_path, mode="w", header=header) as writer:
            # 从文件1读取当前块的点
            # 步骤2: 逐个文件处理
            for i, las_path in enumerate(tqdm(las_paths, desc="Processing files")):
                with laspy.open(las_path) as reader:
                    for points in reader.chunk_iterator(1_000_000):
                        mask = laspy_voxel_max_sampling(points, 0.1)
                        points_ = points[mask]
                        writer.write_points(points_)

    print(f"处理完成！输出文件: {output_path}")




def point_merge1(las_paths, output_path):
    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    chunk_size = 500  # 500米

    # ============= 遍历读取las =============
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    for las_path in las_paths:
        with laspy.open(las_path) as reader:
            # 合并多个文件的范围
            min_x = min(reader.header.mins[0], min_x)
            max_x = max(reader.header.maxs[0], max_x)
            min_y = min(reader.header.mins[1], min_y)
            max_y = max(reader.header.maxs[1], max_y)

    # 确保坐标范围有适当的缓冲区
    buffer = chunk_size * 0.1
    min_x -= buffer
    max_x += buffer
    min_y -= buffer
    max_y += buffer

    # 计算分块网格
    x_blocks = int(np.ceil((max_x - min_x) / chunk_size))
    y_blocks = int(np.ceil((max_y - min_y) / chunk_size))

    # 获取点格式并准备输出
    with laspy.open(las_paths[0]) as reader0:
        # 获取点格式并准备输出
        header = reader0.header
        point_format = reader0.header.point_format

    # 逐块处理各个点云
    with laspy.open(output_path, mode="w", header=header) as writer:
        #  步骤1: 逐区域块处理
        with tqdm(total=x_blocks * y_blocks, desc="处理空间分块") as pbar:
            for x_idx in range(x_blocks):
                for y_idx in range(y_blocks):
                    print(str(x_idx * y_blocks + y_idx) + "/" + str(x_blocks * y_blocks))
                    # 计算当前块范围
                    block_min_x = min_x + x_idx * chunk_size
                    block_max_x = min_x + (x_idx + 1) * chunk_size
                    block_min_y = min_y + y_idx * chunk_size
                    block_max_y = min_y + (y_idx + 1) * chunk_size

                    # 存储当前范围内点
                    block_points = create_las(point_format=point_format).points

                    # 步骤2: 逐个文件处理
                    for i, las_path in enumerate(tqdm(las_paths, desc="Processing files")):
                    # for i, las_path in enumerate(las_paths):
                        with laspy.open(las_path) as reader:
                            min_x_temp = reader.header.mins[0]
                            max_x_temp = reader.header.maxs[0]
                            min_y_temp = reader.header.mins[1]
                            max_y_temp = reader.header.maxs[1]

                            # 判断数据区域是否重合
                            if (min_x_temp <= block_max_x) and (max_x_temp >= block_min_x) and (min_y_temp <= block_max_y) and (max_y_temp >= block_min_y):
                                for points in reader.chunk_iterator(1_000_000):
                                    mask = (points.x >= block_min_x) & (points.x <= block_max_x) & \
                                           (points.y >= block_min_y) & (points.y <= block_max_y)

                                    if np.any(mask):
                                        filtered_points = points[mask]
                                        # 直接追加到PointRecord
                                        if len(block_points) == 0:
                                            block_points = filtered_points
                                        else:
                                            # 高效合并PointRecord
                                            block_points.array = np.hstack([block_points.array, filtered_points.array])
                            else:
                                pass
                    # r如果有数据的话：
                    if len(block_points) != 0:
                        mask = laspy_voxel_max_sampling(block_points, 0.3)
                        block_points_ = block_points[mask]
                        writer.write_points(block_points_)

    print(f"处理完成！输出文件: {output_path}")







if __name__ == '__main__':
    # # 打开txt文件
    names = ["points" + str(i) for i in range(0, 11)]
    point_paths = ["/home/dshare/06Test/LiDAR/America/losanj/{}.laz".format(name) for name in names]
    # names = ["all-losangle0-10", "all-losangle11-19"]
    # point_paths = ["/home/dshare/06Test/LiDAR/America/losanj/{}.laz".format(name) for name in names]
    point_save_name = "/home/dshare/06Test/LiDAR/America/losanj/all-losangle0-11_new.laz"
    # las = point_weiyi(point_paths, point_save_name)
    # las = point_merge(point_paths, point_save_name)
    las = point_merge1(point_paths, point_save_name)
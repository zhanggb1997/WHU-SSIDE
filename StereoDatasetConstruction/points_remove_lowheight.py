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
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from points_process import points_Z_IQR_filter, las_voxel_max_sampling, SelfLas, laspy_voxel_max_sampling


def point_remove_height1(las_paths, output_path):
    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    chunk_size = 200  # 500米

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
                        #  ============= 筛除las中噪声值 ===============
                        classes = block_points.classification.copy()
                        noise_mask = np.isin(classes, [7, 14, 18, 19, 20])  # 噪声

                        # #  ============= 筛除las中噪声值 ===============
                        valid_classes = ~noise_mask
                        filtered_classes = classes[valid_classes]
                        filtered_points = block_points[valid_classes]
                        # writer.write_points(filtered_points)

                        # # #  ============= 筛除las中植被下的地面点 ===============
                        veg_grd_mask = remove_grd_pts(filtered_classes, filtered_points, [3, 4, 5], [1, 2], 3.0)  # 过滤植被的地面值
                        veg_veg_mask = remove_hei_pts(filtered_classes, filtered_points, [3, 4, 5], [3, 4, 5], 6.0, 10., 5)  # 过滤植被高差过高值
                        veg_veh_mask = remove_hei_pts(filtered_classes, filtered_points, [5], [5], 6.0, 8., 5)  # 过滤hei植被高差过高值
                        wat_unc_mask = remove_grd_pts(filtered_classes, filtered_points, [9], [1], 5.0)  # 过滤水面的未分类值

                        # #  ============= 筛除las中低植被\水域 ===============
                        lowveg_mask_ = np.isin(filtered_classes, [3])
                        lowveg_mask = ~lowveg_mask_
                        midveg_mask_ = np.isin(filtered_classes, [4])
                        midveg_mask = ~midveg_mask_
                        wat02_mask_ = np.isin(filtered_classes, [9])
                        wat02_mask = ~wat02_mask_
                        final_mask = veg_grd_mask & veg_veg_mask & veg_veh_mask & wat_unc_mask & lowveg_mask & midveg_mask & wat02_mask
                        filtered_points_ = filtered_points[final_mask]
                        writer.write_points(filtered_points_)

                        # mask = remove_hei_pts(filtered_classes, filtered_points, [1, 2], [1, 2], 6.0, 3., 10)  # 过滤植被高差过高值
                        # block_points_ = block_points[mask]
                        # writer.write_points(block_points_)


    print(f"处理完成！输出文件: {output_path}")



def remove_hei_pts(classes, points, refern_cls=[5], remove_cls=[5], rem_r=4., hei_r=15., hei_n=3):
    ref_mask = np.isin(classes, refern_cls)
    rem_mask = np.isin(classes, remove_cls)
    ref_hei = points[ref_mask].z
    rem_hei = points[rem_mask].z
    ref_pts = np.vstack((points[ref_mask].x, points[ref_mask].y)).T
    rem_pts = np.vstack((points[rem_mask].x, points[rem_mask].y)).T
    rem_indices = np.where(rem_mask)[0]

    # 3. 创建参考KDTree（使用随机采样优化）
    veg_tree = KDTree(ref_pts)
    to_remove = np.zeros(len(rem_pts), dtype=bool)

    # 分块处理策略（优化内存使用）
    block_size = 10_000_000
    processed = 0

    while processed < len(rem_pts):
        end_idx = min(processed + block_size, len(rem_pts))
        block_points = rem_pts[processed:end_idx]

        # 查询每个点半径内的植被点
        indices_list = veg_tree.query_ball_point(block_points, rem_r, workers=-1)

        # 标记有植被邻近的点
        for i, indices in enumerate(indices_list):
            low_sum = 0
            resion_sum = 0
            if len(indices) > 0:
                for indi in indices:
                    if abs(ref_hei[indi] - rem_hei[i]) > hei_r:
                    # if ref_hei[indi] - rem_hei[i] < hei_r:
                        low_sum += 1
                    else:
                        resion_sum += 1
                if low_sum >= hei_n and resion_sum <= hei_n:
                    to_remove[processed + i] = True

        processed = end_idx

    # 最终确定mask
    final_mask = np.ones(len(points), dtype=bool)
    remove_indices = rem_indices[to_remove]
    final_mask[remove_indices] = False

    return final_mask




def remove_grd_pts(classes, points, refern_cls=[3, 4, 5], remove_cls=[1, 2], rem_r=4.):
    ref_mask = np.isin(classes, refern_cls)
    rem_mask = np.isin(classes, remove_cls)
    ref_pts = np.vstack((points[ref_mask].x, points[ref_mask].y)).T
    rem_pts = np.vstack((points[rem_mask].x, points[rem_mask].y)).T
    rem_indices = np.where(rem_mask)[0]

    # 3. 创建参考KDTree（使用随机采样优化）
    veg_tree = KDTree(ref_pts)
    to_remove = np.zeros(len(rem_pts), dtype=bool)

    # 分块处理策略（优化内存使用）
    block_size = 100000
    processed = 0

    while processed < len(rem_pts):
        end_idx = min(processed + block_size, len(rem_pts))
        block_points = rem_pts[processed:end_idx]

        # 查询每个点半径内的植被点
        indices_list = veg_tree.query_ball_point(block_points, rem_r, workers=-1)

        # 标记有植被邻近的点
        for i, indices in enumerate(indices_list):
            if len(indices) > 0:
                to_remove[processed + i] = True

        processed = end_idx

    # 最终确定mask
    final_mask = np.ones(len(points), dtype=bool)
    remove_indices = rem_indices[to_remove]
    final_mask[remove_indices] = False

    return final_mask



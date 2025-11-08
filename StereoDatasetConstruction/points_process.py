'''
Project    : RSDeploy
FileName   : points_process .py
CreateTime : 2025/2/13 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import gc
import math
import os.path
import random
from copy import deepcopy

import cv2
import laspy
import numpy as np
import pyproj
import rasterio
import srtm4
from pyproj import CRS, Transformer

# 读取LAS文件
# # 获取头信息
# header = las.header
#
# # 输出基本点云信息
# print(f"点数量: {header.point_count}")
# print(f"范围 X: {header.x_min} - {header.x_max}")
# print(f"范围 Y: {header.y_min} - {header.y_max}")
# print(f"范围 Z: {header.z_min} - {header.z_max}")
#
# # 尝试解析坐标参考系统（CRS）
# crs = None
# wkt = None
#
# # 检查头中的CRS信息（LASpy 2.4+）
# if hasattr(header, "parse_crs"):
#     crs = header.parse_crs()
#     if crs:
#         print("\n坐标参考系统（CRS）信息:")
#         print(crs)
#
# # 如果没有找到，检查VLR中的WKT信息
# if not crs:
#     for vlr in header.vlrs:
#         # LAS 1.4使用WKT记录（ID 2112）
#         if vlr.record_id == 2112:
#             wkt = vlr.strings[0]
#             crs = CRS.from_wkt(wkt)
#             break
#         # 或检查GeoKeyDirectory（ID 34735）
#         elif vlr.record_id == 34735:
#             # 需要进一步解析GeoKeys获取EPSG代码
#             pass
#
# # 如果仍未找到，检查.prj文件
# if not crs:
#     prj_file = "your_file.prj"
#     try:
#         with open(prj_file, 'r') as f:
#             wkt = f.read()
#             crs = CRS.from_wkt(wkt)
#             print("\n从.prj文件读取CRS:")
#     except FileNotFoundError:
#         print("\n警告：未找到CRS信息。")
#
# # 输出CRS详情
# if crs:
#     print(f"CRS名称: {crs.name}")
#     print(f"坐标类型: {'地理坐标系' if crs.is_geographic else '投影坐标系'}")
#     print(f"EPSG代码: {crs.to_epsg() if crs.to_epsg() else '未知'}")
#
#
#
# # 转换坐标到地理坐标系（示例）
# if crs and not crs.is_geographic:
#     # 创建转换器（假设目标为WGS84，EPSG:4326）
#     target_crs = CRS.from_epsg(4326)
#     transformer = Transformer.from_crs(crs, target_crs)
#
#     # 转换第一个点为例
#     x, y, z = las.x[0], las.y[0], las.z[0]
#     lon, lat = transformer.transform(x, y)
#     print(f"\n第一个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
#     print(f"\n第一个点的x y z: x={x}, {y}, {z}")
#
# else:
#     # 南京地区NJ预先定义的CRS
#     target_crs = pyproj.CRS("EPSG:4326")  # EPGS:4326 代码代表 WGS-84坐标系
#     # 确定对应的UTM区域
#     NJ_center_zone = int(118 / 6) + 31  # UTM区域
#     # 高斯克吕格
#     format = '+proj=tmerc +lat_0=0 +lon_0=' + str(120) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
#     crs_GK = CRS.from_proj4(format)
#     # 定义UTM坐标系（基于WGS84）
#     utm_proj = pyproj.CRS(f"EPSG:326{NJ_center_zone}")  # EPSG:326 代码代表 UTM坐标系
#     # transformer = Transformer.from_crs(utm_proj, wgs84_proj)
#     transformer = Transformer.from_crs(crs_GK , target_crs)
#
#
#
# # 访问点坐标（已应用缩放和偏移）
# print("\n前三点的坐标（X, Y, Z）:")
# for i in range(3):
#     x, y, z = las.x[i], las.y[i], las.z[i]
#     lon, lat = transformer.transform(x, y)
#     print(f"\n第{i + 1}个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
#     print(f"点{i + 1}: {las.x[i]}, {las.y[i]}, {las.z[i]}")
#
#
#
# # 坐标转换
# lats, lons = transformer.transform(las.x, las.y)
# # 创建新文件头
# new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
# new_header.offsets = [0, 0, 0]
# new_header.scales = [1e-7, 1e-7, 0.01]  # 经纬度使用更高精度
#
# # 写入新文件
# new_las = laspy.LasData(header)
# for dim in las.point_format.dimensions:
#     new_las[dim.name] = las[dim.name]
# new_las.x = lons
# new_las.y = lats
# new_las.z = las.z
# # 添加CRS信息（LAS 1.4+）
# if hasattr(new_las.header, "add_crs"):
#     new_las.header.add_crs(target_crs)
#
# new_las.write(output_file)
# print(f"转换完成，文件已保存为：{output_file}")
from rasterio import float32
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from disp_rectify import read_points
from points_rectify import compute_affine_matrix, points_apply_transform, compute_3d_affine_matrix, \
    points_apply_3d_transform, compute_3d_polynomial_matrix, points_apply_3d_polynomial_transform, calculate_residuals, \
    xy_apply_transform

class SelfLas():
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

def get_pixel_value(image_path, lon, lat):
    """
    根据地理坐标 (经度, 纬度) 提取遥感影像的像素值
    """
    with rasterio.open(image_path) as src:
        # # 检查坐标系是否匹配（若需要，进行坐标转换）
        # if src.crs.to_epsg() != 4326:  # 假设输入坐标是 WGS84 (EPSG:4326)
        #     # 使用 pyproj 转换坐标到影像的坐标系
        #     from pyproj import Transformer
        #     transformer = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
        #     x, y = transformer.transform(lon, lat)
        # else:
        #     x, y = lon, lat
        x, y = lon, lat

        # 将地理坐标转换为像素行列号
        row, col = src.index(x, y)

        # 读取像素值（支持多波段）
        value = src.read(1, window=((row, row + 1), (col, col + 1)))  # 读取第一个波段
        return value[0][0]


def remove_grd_pts(classes, points, refern_cls=[3, 4, 5], remove_cls=[1, 2], rem_r=4.):
    ref_mask = np.isin(classes, refern_cls)
    rem_mask = np.isin(classes, remove_cls)
    ref_pts = points[ref_mask][:, :2]
    rem_pts = points[rem_mask][:, :2]
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


def remove_hei_pts(classes, points, refern_cls=[5], remove_cls=[5], rem_r=4., hei_r=20., hei_n=3):
    ref_mask = np.isin(classes, refern_cls)
    rem_mask = np.isin(classes, remove_cls)
    ref_hei = points[ref_mask][:, 2]
    rem_hei = points[rem_mask][:, 2]
    ref_pts = points[ref_mask][:, :2]
    rem_pts = points[rem_mask][:, :2]
    rem_indices = np.where(rem_mask)[0]

    # 3. 创建参考KDTree（使用随机采样优化）
    veg_tree = KDTree(ref_pts)
    to_remove = np.zeros(len(rem_pts), dtype=bool)

    # 分块处理策略（优化内存使用）
    block_size = 1_000_000
    processed = 0

    while processed < len(rem_pts):
        end_idx = min(processed + block_size, len(rem_pts))
        block_points = rem_pts[processed:end_idx]

        # 查询每个点半径内的植被点
        indices_list = veg_tree.query_ball_point(block_points, rem_r, workers=-1)

        # 标记有植被邻近的点
        for i, indices in enumerate(indices_list):
            low_sum = 0
            if len(indices) > 0:
                for indi in indices:
                    if ref_hei[indi] - rem_hei[i] > hei_r:
                    # if ref_hei[indi] - rem_hei[i] < hei_r:
                        low_sum += 1
                if low_sum > hei_n:
                    to_remove[processed + i] = True

        processed = end_idx

    # 最终确定mask
    final_mask = np.ones(len(points), dtype=bool)
    remove_indices = rem_indices[to_remove]
    final_mask[remove_indices] = False

    return final_mask


def remove_hei_pts_losangle(points, rem_r=4., hei_r=10., hei_n=3):
    ref_hei = points.z
    rem_hei = points.z
    ref_pts = np.vstack((points.x, points.y)).T
    rem_pts = np.vstack((points.x, points.y)).T

    # 创建参考KDTree（使用随机采样优化）
    veg_tree = KDTree(ref_pts)
    to_remove = np.zeros(len(rem_pts), dtype=bool)

    # 分块处理策略（优化内存使用）
    block_size = 1_000_000
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
                        low_sum += 1
                    else:
                        resion_sum += 1
                if low_sum > hei_n and resion_sum <= hei_n:
                    to_remove[processed + i] = True

        processed = end_idx

    # 最终确定mask
    final_mask = ~to_remove

    return final_mask


def remove_unc_pts(classes, points, refern_cls=[6], remove_cls=[1], rem_r=4., hei_r=0.):
    ref_mask = np.isin(classes, refern_cls)
    rem_mask = np.isin(classes, remove_cls)
    ref_hei = points[ref_mask][:, 2]
    rem_hei = points[rem_mask][:, 2]
    ref_pts = points[ref_mask][:, :2]
    rem_pts = points[rem_mask][:, :2]
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
                for indi in indices:
                    if rem_hei[i] - ref_hei[indi] < hei_r:
                        to_remove[processed+i] = True
                        break

        processed = end_idx

    # 最终确定mask
    final_mask = np.ones(len(points), dtype=bool)
    remove_indices = rem_indices[to_remove]
    final_mask[remove_indices] = False

    return final_mask


def LasInit(las_paths):

    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    all_las = []

    # # 降采样体素尺寸
    voxel_size = 1

    # 读取所有数据，初步筛除高程不符合区域
    for las_path in tqdm(las_paths):
        #  ============= 直接读取las ===============
        # 读取所有las
        las = laspy.read(las_path)
        # all_las.append(las)
    #     #### LosAngles
    #     # 基于头信息，创建合并后的LAS对象
    #     merged_points_x = np.array(las.x).astype(np.float64)
    #     merged_points_y = np.array(las.y).astype(np.float64)
    #     merged_points_z = np.array(las.z).astype(np.float32)
    #
    #     low_mask = merged_points_z < 1
    #     low_mask_ = ~low_mask
    #     hei_mask = merged_points_z > 330
    #     hei_mask_ = ~hei_mask
    #
    #     mask_ = low_mask_ & hei_mask_
    #
    #     filter_las1 = SelfLas()
    #     filter_las1.x = merged_points_x[mask_]
    #     filter_las1.y = merged_points_y[mask_]
    #     filter_las1.z = merged_points_z[mask_]
    #
    #     # error_mask = remove_hei_pts_losangle(filter_las1, 6.0, 5., 6)  # 过滤植被高差过高值
    #     #
    #     # filter_las2 = SelfLas()
    #     # filter_las2.x = filter_las1.x[error_mask]
    #     # filter_las2.y = filter_las1.y[error_mask]
    #     # filter_las2.z = filter_las1.z[error_mask]
    #
    #     all_las.append(filter_las1)
    #
    # # 基于头信息，创建合并后的LAS对象
    # merged_points_x = np.concatenate([las.x for las in all_las]).astype(np.float64)
    # merged_points_y = np.concatenate([las.y for las in all_las]).astype(np.float64)
    # merged_points_z = np.concatenate([las.z for las in all_las]).astype(np.float32)
    # merged_las = SelfLas()
    # merged_las.x = merged_points_x
    # merged_las.y = merged_points_y
    # merged_las.z = merged_points_z
    #
    # return merged_las


        # points = np.vstack((las.x, las.y, las.z)).T
        # classes = las.classification.copy()

        #  ============= cook ===============


        #
        # #  ============= 筛除las中噪声值 ===============
        # noise_mask = np.isin(classes, [7, 18, 20])  # 极低噪声、高噪声、20
        # valid_classes = ~noise_mask
        # filtered_classes = classes[valid_classes]
        # filtered_points = points[valid_classes]

        # # #  ============= 筛除las中植被下的地面点 ===============
        # veg01_mask = remove_grd_pts(filtered_classes, filtered_points, [3, 4, 5], [1, 2], 4.0)  # 过滤植被的地面值
        # veg02_mask = remove_hei_pts(filtered_classes, filtered_points, [3, 4, 5], [3, 4, 5], 6.0, 10., 5)  # 过滤植被高差过高值
        # # veg03_mask = remove_grd_pts(filtered_classes, filtered_points, [5], [4, 3], 4.0)  # 过滤植被附近的中等植被值
        # veg03_mask = remove_hei_pts(filtered_classes, filtered_points, [5], [5], 6.0, 8., 5)  # 过滤植被高差过高值
        # # veg02_mask = remove_hei_pts(filtered_classes, filtered_points, [4], [4], 6.0, -6., 5)  # 过滤植被高差过高值
        # bud01_mask = remove_grd_pts(filtered_classes, filtered_points, [6], [2], 2.0)  # 过滤建筑的地面值
        # bud02_mask = remove_unc_pts(filtered_classes, filtered_points, [6], [1], 3.0, 1.0)  # 过滤建筑的未分类值
        # wat01_mask = remove_grd_pts(filtered_classes, filtered_points, [9], [1], 5.0)  # 过滤水面的未分类值
        #
        # # #  ============= 筛除las中低植被 ===============
        # lowveg_mask_ = np.isin(filtered_classes, [3])
        # lowveg_mask = ~lowveg_mask_
        # midveg_mask_ = np.isin(filtered_classes, [4])
        # midveg_mask = ~midveg_mask_
        # # heiveg_mask_ = np.isin(filtered_classes, [5])
        # # heiveg_mask = ~heiveg_mask_
        #
        # #  ============= 筛除las中水域 ===============
        # wat02_mask_ = np.isin(filtered_classes, [9])
        # wat02_mask = ~wat02_mask_
        #
        # # final_mask = wat01_mask & wat02_mask & veg01_mask & veg02_mask & bud01_mask & bud02_mask & lowveg_mask & heiveg_mask
        # final_mask = wat01_mask & wat02_mask & veg01_mask & veg02_mask & veg03_mask & bud01_mask & bud02_mask & lowveg_mask & midveg_mask
        # # final_mask = wat01_mask & wat02_mask & veg01_mask & veg02_mask & bud01_mask & bud02_mask & lowveg_mask & veg03_mask
        # # final_mask = veg01_mask & build_mask & lowveg_mask & midveg_mask
        # # veg_mask = np.isin(filtered_classes, [3, 4, 5])
        # # grd_mask = np.isin(filtered_classes, [1, 2])
        # # veg_pts = filtered_points[veg_mask][:, :2]
        # # grd_pts = filtered_points[grd_mask][:, :2]
        # # grd_indices = np.where(grd_mask)[0]
        # #
        # # # 3. 创建植被KDTree（使用随机采样优化）
        # # veg_tree = KDTree(veg_pts)
        # # to_remove = np.zeros(len(grd_pts), dtype=bool)
        # #
        # # # 分块处理策略（优化内存使用）
        # # block_size = 100000
        # # processed = 0
        # #
        # # while processed < len(grd_pts):
        # #     end_idx = min(processed + block_size, len(grd_pts))
        # #     block_points = grd_pts[processed:end_idx]
        # #
        # #     # 查询每个点半径内的植被点
        # #     indices_list = veg_tree.query_ball_point(block_points, 3.0, workers=-1)
        # #
        # #     # 标记有植被邻近的点
        # #     for i, indices in enumerate(indices_list):
        # #         if len(indices) > 0:
        # #             to_remove[processed + i] = True
        # #
        # #     processed = end_idx
        # #
        # # # 最终确定mask
        # # final_mask = np.ones(len(filtered_points), dtype=bool)
        # # remove_indices = grd_indices[to_remove]
        # # final_mask[remove_indices] = False
        #
        # #  ============= las转为SelfLas ===============
        # las_f = SelfLas()
        # las_f.x = np.array(filtered_points[:,0])[final_mask]
        # las_f.y = np.array(filtered_points[:,1])[final_mask]
        # las_f.z = np.array(filtered_points[:,2])[final_mask]
        # las_f = SelfLas()
        # las_f.x = np.array(filtered_points[:,0])
        # las_f.y = np.array(filtered_points[:,1])
        # las_f.z = np.array(filtered_points[:,2])
        las_f = SelfLas()
        las_f.x = np.array(las.x)
        las_f.y = np.array(las.y)
        las_f.z = np.array(las.z)
        all_las.append(las_f)

        # #  ============= 筛除las中异常高程 ===============
        # mask_z = points_Z_IQR_filter(las, iqr_scale=0.02)
        # las_f = SelfLas()
        # las_f.x = np.array(las.x)[mask_z]
        # las_f.y = np.array(las.y)[mask_z]
        # las_f.z = np.array(las.z)[mask_z]

        # # #  ============= las体素格降采样 ===============
        # las_d = las_voxel_max_sampling(las_f, voxel_size)
        #
        # all_las.append(las_d)
    #
    #
    # 基于头信息，创建合并后的LAS对象
    merged_points_x = np.concatenate([las.x for las in all_las]).astype(np.float64)
    merged_points_y = np.concatenate([las.y for las in all_las]).astype(np.float64)
    merged_points_z = np.concatenate([las.z for las in all_las]).astype(np.float32)
    merged_las = SelfLas()
    merged_las.x = merged_points_x
    merged_las.y = merged_points_y
    merged_las.z = merged_points_z

    # #  ============= 筛除las中异常高程 ===============
    # mask_z = points_Z_IQR_filter(merged_las, iqr_scale=0.001)
    # filtered_las = SelfLas()
    # filtered_las.x = merged_las.x[mask_z]
    # filtered_las.y = merged_las.y[mask_z]
    # filtered_las.z = merged_las.z[mask_z]

    del all_las, merged_points_x, merged_points_y, merged_points_z
    gc.collect()

    return merged_las
# # 基于头信息，创建合并后的LAS对象
#     merged_las = laspy.LasData(all_las[0].header)
#
#     # 合并点数据
#     merged_points_array = np.concatenate([las.points.array for las in all_las])
#     merged_las.points.array = merged_points_array
#     merged_points_x = np.concatenate([las.xyz[:, 0] for las in all_las]).astype(np.float32)
#     merged_points_y = np.concatenate([las.xyz[:, 1] for las in all_las]).astype(np.float32)
#     merged_points_z = np.concatenate([las.xyz[:, 2] for las in all_las]).astype(np.float32)
#     merged_las.x = merged_points_x
#     merged_las.y = merged_points_y
#     merged_las.z = merged_points_z
#
#     # 更新信息
#     merged_header = merged_las.header
#     merged_header.mins = [
#         min([las.header.x_min for las in all_las]),
#         min([las.header.y_min for las in all_las]),
#         min([las.header.z_min for las in all_las]),
#     ]
#     merged_header.maxs = [
#         max([las.header.x_max for las in all_las]),
#         max([las.header.y_max for las in all_las]),
#         max([las.header.z_max for las in all_las]),
#     ]
#     merged_header.x_min = min([las.header.x_min for las in all_las])
#     merged_header.x_max = max([las.header.x_max for las in all_las])
#     merged_header.y_min = min([las.header.y_min for las in all_las])
#     merged_header.y_max = max([las.header.y_max for las in all_las])
#     merged_header.z_min = min([las.header.z_min for las in all_las])
#     merged_header.z_max = max([las.header.z_max for las in all_las])
#
#     # 复制额外属性
#     for dim in all_las[0].point_format.dimensions:
#         if dim.name not in ["X", "Y", "Z"]:
#             merged_data = np.concatenate([las[dim.name] for las in all_las])
#             merged_las[dim.name] = merged_data
#
#     #  ============= 筛除las中异常高程 ===============
#     # # 筛选Z ≤ 800的点
#     # mask_z = las.z <= 900
#     # Z-Score
#     # _, mask_z = points_Z_zscore_filter(las, z_threshold=3.0)
#     # IQR
#     _, mask_z = points_Z_IQR_filter(merged_las, iqr_scale=1.0)
#     filtered_points = merged_las.points[mask_z]
#
#     # 创建新文件并写入筛选后的点
#     filtered_las = laspy.LasData(merged_las.header)
#     filtered_las.points = filtered_points
#
#     las_res = filtered_las
#
#     del all_las, merged_las
#     gc.collect()
#
#     return las_res

# def point_localization(las, save_path):
#     # 尝试解析坐标参考系统（CRS）
#     crs = None
#     wkt = None
#
#     # 检查头中的CRS信息（LASpy 2.4+）
#     if hasattr(header, "parse_crs"):
#         crs = header.parse_crs()
#         if crs:
#             print("\n坐标参考系统（CRS）信息:")
#             print(crs)
#
#     # 如果没有找到，检查VLR中的WKT信息
#     if not crs:
#         for vlr in header.vlrs:
#             # LAS 1.4使用WKT记录（ID 2112）
#             if vlr.record_id == 2112:
#                 wkt = vlr.strings[0]
#                 crs = CRS.from_wkt(wkt)
#                 break
#             # 或检查GeoKeyDirectory（ID 34735）
#             elif vlr.record_id == 34735:
#                 # 需要进一步解析GeoKeys获取EPSG代码
#                 pass
#
#     # 如果仍未找到，检查.prj文件
#     if not crs:
#         prj_file = "your_file.prj"
#         try:
#             with open(prj_file, 'r') as f:
#                 wkt = f.read()
#                 crs = CRS.from_wkt(wkt)
#                 print("\n从.prj文件读取CRS:")
#         except FileNotFoundError:
#             print("\n警告：未找到CRS信息。")
#
#     # 输出CRS详情
#     if crs:
#         print(f"CRS名称: {crs.name}")
#         print(f"坐标类型: {'地理坐标系' if crs.is_geographic else '投影坐标系'}")
#         print(f"EPSG代码: {crs.to_epsg() if crs.to_epsg() else '未知'}")
#
#     # 转换坐标到地理坐标系（示例）
#     if crs and not crs.is_geographic:
#         # 创建转换器（假设目标为WGS84，EPSG:4326）
#         target_crs = CRS.from_epsg(4326)
#         transformer = Transformer.from_crs(crs, target_crs)
#
#         # 转换第一个点为例
#         x, y, z = las.x[0], las.y[0], las.z[0]
#         lon, lat = transformer.transform(x, y)
#         print(f"\n第一个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
#         print(f"\n第一个点的x y z: x={x}, {y}, {z}")
#
#     else:
#         # 南京地区NJ预先定义的CRS
#         target_crs = pyproj.CRS("EPSG:4326")  # EPGS:4326 代码代表 WGS-84坐标系
#         # 确定对应的UTM区域
#         NJ_center_zone = int(118 / 6) + 31  # UTM区域
#         # 高斯克吕格
#         format = '+proj=tmerc +lat_0=0 +lon_0=' + str(120) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
#         crs_GK = CRS.from_proj4(format)
#         # 定义UTM坐标系（基于WGS84）
#         utm_proj = pyproj.CRS(f"EPSG:326{NJ_center_zone}")  # EPSG:326 代码代表 UTM坐标系
#         # transformer = Transformer.from_crs(utm_proj, wgs84_proj)
#         transformer = Transformer.from_crs(crs_GK, target_crs)
#
#     # 访问点坐标（已应用缩放和偏移）
#     print("\n前三点的坐标（X, Y, Z）:")
#     for i in range(3):
#         x, y, z = las.x[i], las.y[i], las.z[i]
#         lon, lat = transformer.transform(x, y)
#         print(f"\n第{i + 1}个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
#         print(f"点{i + 1}: {las.x[i]}, {las.y[i]}, {las.z[i]}")
#
#     # 坐标转换
#     lats, lons = transformer.transform(las.x, las.y)
#     # 创建新文件头
#     new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
#     new_header.offsets = [0, 0, 0]
#     new_header.scales = [1e-7, 1e-7, 0.01]  # 经纬度使用更高精度
#
#     # 写入新文件
#     new_las = laspy.LasData(header)
#     for dim in las.point_format.dimensions:
#         new_las[dim.name] = las[dim.name]
#     new_las.x = lons
#     new_las.y = lats
#     new_las.z = las.z
#     # 添加CRS信息（LAS 1.4+）
#     if hasattr(new_las.header, "add_crs"):
#         new_las.header.add_crs(target_crs)
#
#     new_las.write(save_path)
#     print(f"转换完成，文件已保存为：{save_path}")
#
#


def point_localization(las, save_path=None):
    # America
    # 创建转换器（假设目标为WGS84，EPSG:4326）
    target_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs("EPSG:6455", target_crs)
    # transformer = Transformer.from_crs("EPSG:32610", target_crs)  #losangle

    # 坐标转换
    lats, lons = transformer.transform(las.x, las.y)
    # # 获取正高和椭球高的差异
    path_emg08 = "/home/dshare/06Test/us_nga_egm2008_1.tif"
    eval_dsip = get_pixel_value(path_emg08, np.array(lons)[-1], np.array(lats)[-1])
    #
    # # 高程英尺转为米，并添加偏移量处理
    heis = 0.3048 * np.array(las.z) + eval_dsip

    # heis = np.array(las.z)

    new_las = SelfLas()
    new_las.x = lons.astype(np.float64)
    new_las.y = lats.astype(np.float64)
    new_las.z = heis.astype(np.float32)

    # 如果有保存地址的话
    if save_path:
        # 输出dsm.tif
        point_2_dsm(new_las, save_path, lt_x=np.min(np.array(lons)), lt_y=np.min(np.array(lats)), res_x=0.00001, res_y=-0.00001, crs=target_crs)

    # 返回信息
    return new_las

    #
    # # las头信息
    # header = las.header
    #
    # # 尝试解析坐标参考系统（CRS）
    # crs = None
    #
    # # 检查头中的CRS信息（LASpy 2.4+）
    # if hasattr(header, "parse_crs"):
    #     crs = header.parse_crs()
    #     if crs:
    #         print("\n坐标参考系统（CRS）信息:")
    #         print(crs)
    #
    # # 如果没有找到，检查VLR中的WKT信息
    # if not crs:
    #     for vlr in header.vlrs:
    #         # LAS 1.4使用WKT记录（ID 2112）
    #         if vlr.record_id == 2112:
    #             wkt = vlr.strings[0]
    #             crs = CRS.from_wkt(wkt)
    #             break
    #         # 或检查GeoKeyDirectory（ID 34735）
    #         elif vlr.record_id == 34735:
    #             # 需要进一步解析GeoKeys获取EPSG代码
    #             pass
    #
    # # 如果仍未找到，检查.prj文件
    # if not crs:
    #     prj_file = "your_file.prj"
    #     try:
    #         with open(prj_file, 'r') as f:
    #             wkt = f.read()
    #             crs = CRS.from_wkt(wkt)
    #             print("从.prj文件读取CRS:")
    #     except FileNotFoundError:
    #         print("警告：未找到CRS信息。")
    #
    # # 输出CRS详情
    # if crs:
    #     print(f"CRS名称: {crs.name}")
    #     print(f"坐标类型: {'地理坐标系' if crs.is_geographic else '投影坐标系'}")
    #     print(f"EPSG代码: {crs.to_epsg() if crs.to_epsg() else '未知'}")
    #
    # # 转换坐标到地理坐标系（示例）
    # if crs.to_epsg() and not crs.is_geographic:
    #     # 创建转换器（假设目标为WGS84，EPSG:4326）
    #     target_crs = CRS.from_epsg(4326)
    #     transformer = Transformer.from_crs(crs, target_crs)
    #
    #     # # 转换第一个点为例
    #     # x, y, z = las.x[0], las.y[0], las.z[0]
    #     # lon, lat = transformer.transform(x, y)
    #     # print(f"\n第一个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
    #     # print(f"\n第一个点的x y z: x={x}, {y}, {z}")
    #
    # elif crs and not crs.is_geographic:  # America
    #     # 创建转换器（假设目标为WGS84，EPSG:4326）
    #     target_crs = CRS.from_epsg(4326)
    #     transformer = Transformer.from_crs("EPSG:6455", target_crs)
    #
    # else:
    #     # 南京地区NJ预先定义的CRS
    #     target_crs = pyproj.CRS("EPSG:4326")  # EPGS:4326 代码代表 WGS-84坐标系
    #     # 确定对应的UTM区域
    #     NJ_center_zone = int(118 / 6) + 31  # UTM区域
    #     # 高斯克吕格
    #     format = '+proj=tmerc +lat_0=0 +lon_0=' + str(120) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    #     crs_GK = CRS.from_proj4(format)
    #     # # 定义UTM坐标系（基于WGS84）
    #     # utm_proj = pyproj.CRS(f"EPSG:326{NJ_center_zone}")  # EPSG:326 代码代表 UTM坐标系
    #     # transformer = Transformer.from_crs(utm_proj, wgs84_proj)
    #     transformer = Transformer.from_crs(crs_GK, target_crs)
    #     # transformer = Transformer.from_crs("EPSG:6455", target_crs)
    #
    # # # 访问点坐标（已应用缩放和偏移）
    # # print("\n前三点的坐标（X, Y, Z）:")
    # # for i in range(3):
    # #     x, y, z = las.x[i], las.y[i], las.z[i]
    # #     lon, lat = transformer.transform(x, y)
    # #     print(f"\n第{i + 1}个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
    # #     print(f"点{i + 1}: {las.x[i]}, {las.y[i]}, {las.z[i]}")
    #
    # # 坐标转换
    # lats, lons = transformer.transform(las.x, las.y)
    # # 获取正高和椭球高的差异
    # path_emg08 = "/home/dshare/06Test/us_nga_egm2008_1.tif"
    # eval_dsip = get_pixel_value(path_emg08, np.array(lons)[-1], np.array(lats)[-1])
    #
    # # 高程英尺转为米，并添加偏移量处理
    # heis = 0.3048 * np.array(las.z) + eval_dsip
    #
    # # 创建新文件头
    # new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    #
    # # 写入新文件
    # new_las = laspy.LasData(new_header)
    # new_las.header.offsets = [0, 0, 0]
    # new_las.points.offsets = [0, 0, 0]
    # new_las.header.scales = [1e-7, 1e-7, 0.001]  # 经纬度使用更高精度
    # new_las.points.scales = [1e-7, 1e-7, 0.001]  # 经纬度使用更高精度
    # for dim in las.point_format.dimensions:
    #     new_las[dim.name] = las[dim.name]
    # new_las.xyz = np.array([lons, lats, heis]).astype(np.float64).transpose(1,0)
    # # new_las.x = np.array(lons).astype(np.float64)
    # # new_las.y = np.array(lats).astype(np.float64)
    # # new_las.z = las.z
    # # 添加CRS信息（LAS 1.4+）
    # if hasattr(new_las.header, "add_crs"):
    #     new_las.header.add_crs(target_crs)
    #
    # # # 如果有保存地址的话
    # # if save_path:
    # #     new_las.write(save_path)
    # #     print(f"转换完成，文件已保存为：{save_path}")
    #
    # # 输出dsm.tif
    # point_2_dsm(new_las, save_path, lt_x=np.min(np.array(lons)), lt_y=np.min(np.array(lats)), res_x=0.00001, res_y=-0.00001, crs=target_crs)
    #
    # # 返回信息
    # return new_las
    #

def get_las_bbox(las):

    las_x_min = np.min(las.x)
    las_x_max = np.max(las.x)
    las_y_min = np.min(las.y)
    las_y_max = np.max(las.y)

    return [
        [las_x_min, las_y_min],
        [las_x_min, las_y_max],
        [las_x_max, las_y_max],
        [las_x_max, las_y_min]
       ]


def point_projection(las, rpcm):

    x_, y_ = rpcm.projection(las.x, las.y, las.z)

    # 创建新文件头
    new_las = SelfLas()

    # 写入新文件
    new_las.x = x_.astype(np.float32)
    new_las.y = y_.astype(np.float32)
    new_las.z = las.z.astype(np.float32)
    return new_las

    #
    # # random_samples = random.sample(range(len(las)), 1000)
    # # z_srtm = srtm4.srtm4(np.array(las.x[random_samples]), np.array(las.y[random_samples]))
    # # z_las  = np.array(las.z[random_samples])
    # # z_shift_mean = np.mean(z_srtm - z_las)
    # # z_ = las.z + z_shift_mean
    # # x_, y_ = rpcm.projection(las.x, las.y, z_)
    #
    # # z_min = np.min(las.z)
    # # z_ = (las.z - z_min) * 0.3546 + z_min
    # # x_, y_ = rpcm.projection(las.x, las.y, z_)
    #
    # x_, y_ = rpcm.projection(las.x, las.y, las.z)
    #
    # # 创建新文件头
    # new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    #
    # # 写入新文件
    # new_las = laspy.LasData(new_header)
    # new_las.header.offsets = [0, 0, 0]
    # new_las.points.offsets = [0, 0, 0]
    # new_las.header.scales = [0.01, 0.01, 0.001]  # 像素坐标系
    # new_las.points.scales = [0.01, 0.01, 0.001]  # 像素坐标系使用更低精度
    # for dim in las.point_format.dimensions:
    #     new_las[dim.name] = las[dim.name]
    # new_las.xyz = np.array([x_, y_, las.z]).astype(np.float32).transpose(1, 0)
    # new_las.x = np.array(x_).astype(np.float32)
    # new_las.y = np.array(y_).astype(np.float32)
    # new_las.z = np.array(las.z).astype(np.float32)
    # return new_las


def save_las(las, las_save_path):
    las.write(las_save_path)
    print(f"文件已保存为：{las_save_path}")


def point_projection_epipolar(las_pixel, epipolar_coeffs):
    pass


def point_2_dsm(las, save_path=None, x_min_max=None, y_min_max=None, lt_x=0, lt_y=0, res_x=1., res_y=1., crs=None):
    if not ((x_min_max) and (y_min_max)):
        x = las.x
        y = las.y
        z = las.z
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
    else:
        x_min, x_max = x_min_max
        y_min, y_max = y_min_max
        x = np.append(np.clip(las.x, x_min, x_max), [x_min, x_max])
        y = np.append(np.clip(las.y, y_min, y_max), [y_min, y_max])
        z = np.append(las.z, [0., 0.])

    if save_path:

        # 创建栅格网格
        if res_x != 1 or res_y != 1:
            cols = abs(int((x_max - x_min)/res_x))
            rows = abs(int((y_max - y_min)/res_y))
        else:
            cols = math.ceil(x_max) - math.floor(x_min)
            rows = math.ceil(y_max) - math.floor(y_min)


        # # 按Z值排序遍历处理
        # dsm = np.zeros((rows, cols), dtype=np.float64)
        # # 将点云分配到栅格（取最大Z值）
        # x_idx = (x - x_min).astype(int)
        # y_idx = (y - y_min).astype(int)  # Y轴不需翻转
        # for i in range(len(x)):
        #     if 0 <= x_idx[i] < cols and 0 <= y_idx[i] < rows:
        #         if z[i] > dsm[y_idx[i], x_idx[i]]:
        #             dsm[y_idx[i], x_idx[i]] = z[i]


        # 使用分箱统计计算每个像素的最大高程
        dsm, col_edges, row_edges, _ = binned_statistic_2d(
            y, x, values=z,
            statistic='max', bins=(rows, cols),
            # range=[[0, cols], [0, rows]]
        )
        # 处理缺失值（无点的像素设为NaN）
        dsm = np.where(np.isnan(dsm), np.nan, dsm)
        # if crs:
        #     dsm = np.flip(dsm, 0)

        # # Create a grid
        # xi, yi = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
        # # Interpolate elevation values
        # dsm = griddata((x, y), z, (xi, yi), method='linear', fill_value=0)  # 线性内插

        # # 自动匹配需要处理为uint8
        # mean_value = np.mean(dsm, where=dsm!=0)
        # std_value = np.std(dsm[dsm!=0])
        # min_value = mean_value - std_value * 2
        # max_value = mean_value + std_value * 2
        # dsm_ = np.clip(dsm, min_value, max_value)
        # dsm_ = ((dsm_ - min_value) / (max_value - min_value) * 255).astype(np.uint8)

        # 手动匹配需要保存为tiff
        assert save_path, print("需要给出dsm输出路径")
        # 定义地理变换参数（需根据实际CRS调整原点）
        transform = from_origin(lt_x, lt_y, res_x, res_y)

        # 写入GeoTIFF
        try:
            with rasterio.open(
                    save_path,
                    "w",
                    driver="GTiff",
                    height=rows,
                    width=cols,
                    count=1,
                    # dtype=dsm.dtype,
                    dtype=float32,
                    crs=crs,  # 指定坐标系,像素坐标系
                    transform=transform,
                    nodata=np.nan
            ) as dst:
                dst.write(dsm, 1)
        except FileExistsError:
            print("dsm.tif已经存在！")
        dsm_ = dsm
    else:
        dsm_ = None

    return dsm_, x_min, y_min


def point_rectify(las, img, match_split=5, save_path_dsm=None, save_path_img=None, read_path_points=None, save_path_affine_matrix=None):
    # 转为DSM
    # _, x_min, y_min = point_2_dsm(las)
    dsm_, x_min, y_min = point_2_dsm(las, save_path_dsm)

    # # # ==== 自动化匹配 ====
    # # sub_imgs = {}
    # # sub_dsms = {}
    # #
    # # # 匹配DSM和img
    # # dsm_xs = np.linspace(0, dsm.shape[1], match_split).astype(int)
    # # dsm_ys = np.linspace(0, dsm.shape[0], match_split).astype(int)
    # # img_xs = np.linspace(x_min, x_min + dsm.shape[1], match_split).astype(int)
    # # img_ys = np.linspace(y_min, y_min + dsm.shape[0], match_split).astype(int)
    # #
    # # for i in range(len(dsm_xs) - 1):
    # #     for j in range(len(dsm_ys) - 1):
    # #         sub_dsms[(dsm_xs[i], dsm_ys[j])] = dsm[dsm_ys[j]: dsm_ys[j+1], dsm_xs[j]: dsm_xs[j+1]]
    # #         sub_imgs[(img_xs[i], img_ys[j])] = img.get_sub_image([img_xs[i], img_ys[j], img_xs[i+1]-img_xs[i], img_xs[j+1]-img_xs[j]])
    # #
    # # matches_good, keyPoint_img, keyPoint_dsm = get_sifts(sub_imgs, sub_dsms)
    # #
    # # for pointdsm in keyPoint_dsm:
    # #     pointdsm.pt = (pointdsm.pt[0] + x_min, pointdsm.pt[1] + y_min)
    # #
    # # good_points = []
    # #
    # # for match_g in matches_good:
    # #     q_id = match_g[0].queryIdx  # 图1中的查询匹配点id
    # #     p_id = match_g[0].trainIdx  # 图2中的对应匹配点id
    # #     keyPimg = keyPoint_img[q_id]  # 图1中的关键点
    # #     keyPdsm = keyPoint_dsm[p_id]  # 图2中的关键点
    # #     x_img, y_img = keyPimg.pt[0], keyPimg.pt[1]  # 图1关键点位置 x,y 像素坐标
    # #     x_dsm, y_dsm = keyPdsm.pt[0], keyPdsm.pt[1]  # 图2关键点位置 x,y 像素坐标
    # #
    # #     good_points.append([x_img, y_img, x_dsm, y_dsm])
    #
    #
    # ==== 手动匹配 ====
    img_ = img.get_sub_image([x_min, y_min, dsm_.shape[1], dsm_.shape[0]])
    cv2.imwrite(save_path_img, img_)

    del dsm_, img_

    if os.path.exists(read_path_points):
        # 读取points
        points = read_points(read_path_points)

        good_points = []

        for point in points:
            # # # 增加高程连接点
            # dsm_z = dsm[int(point[1]), int(point[0])]
            # if dsm_z == 0:
            #     continue
            # good_points.append([point[0]+x_min, abs(point[1])+y_min, dsm_z, point[2]+x_min, abs(point[3])+y_min])
            good_points.append([point[0] + x_min, abs(point[1]) + y_min,
                                point[2] + x_min + 0.5, abs(point[3]) + y_min + 0.5])

        # 带有Z的n阶多项式
        # degree = 3
        # affine_matrix = compute_3d_polynomial_matrix(np.array(good_points), degree)
        # transform_las = points_apply_3d_polynomial_transform(las, affine_matrix, degree)
        # # 带有Z的一阶仿射变换
        # affine_matrix, residuals = compute_3d_affine_matrix(good_points)
        # transform_las = points_apply_3d_transform(las, affine_matrix)
        #  一阶仿射变换
        affine_matrix, residuals = compute_affine_matrix(good_points)

        # 残差计算
        calculate_residuals(affine_matrix, good_points)

        # # 数据保存 仿射变换矩阵
        # with open(save_path_affine_matrix, 'w') as file:
        #     # line = str(affine_matrix[0][0]) + '\t' + str(affine_matrix[0][1]) + '\t' + str(affine_matrix[0][2]) + '\t' + str(affine_matrix[0][3]) + '\n' + \
        #     #        str(affine_matrix[1][0]) + '\t' + str(affine_matrix[1][1]) + '\t' + str(affine_matrix[1][2]) + '\t' + str(affine_matrix[1][3]) + '\n' + \
        #     #        str(affine_matrix[2][0]) + '\t' + str(affine_matrix[2][1]) + '\t' + str(affine_matrix[2][2]) + '\t' + str(affine_matrix[2][3])
        #     line = str(affine_matrix[0][0]) + '\t' + str(affine_matrix[0][1]) + '\t' + str(affine_matrix[0][2]) + '\n' + \
        #            str(affine_matrix[1][0]) + '\t' + str(affine_matrix[1][1]) + '\t' + str(affine_matrix[1][2]) + '\n' + \
        #            str(affine_matrix[2][0]) + '\t' + str(affine_matrix[2][1]) + '\t' + str(affine_matrix[2][2])
        #     file.write(line)

    # # 根据其他的变换矩阵进行变换
    # elif os.path.exists(save_path_affine_matrix):
    #     # 数据读取 仿射变换矩阵
    #     affine_matrix = [
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0]]
    #     with open(save_path_affine_matrix, 'r') as file:
    #         lines = file.readlines()
    #         for line_no, line in enumerate(lines):
    #             affine_matrix[line_no] = [float(a) for a in line.strip().split('\t')]
    # else:
    #     print("仿射变换错误！没有对应匹配信息！")

        # 仿射变换
        transform_las = points_apply_transform(las, affine_matrix)

        dsm_t, x_min_t, y_min_t = point_2_dsm(transform_las, os.path.join(os.path.split(save_path_dsm)[0], os.path.splitext(os.path.basename(save_path_dsm))[0] + "_t.tif"))
        img_t = img.get_sub_image([x_min_t, y_min_t, dsm_t.shape[1], dsm_t.shape[0]])
        cv2.imwrite(os.path.join(os.path.split(save_path_img)[0], os.path.splitext(os.path.basename(save_path_img))[0] + "_t.tif"), img_t)

        del dsm_t, img_t, las

    else:
        transform_las = las

    return transform_las


def las_reproject1L(las, xy_rect1, xx_range1, yy_range1, yy_step1):
    # 原始信息数据
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x1_floor, max_x1_ceil = math.floor(min_x1), math.ceil(max_x1)
    min_y1_floor, max_y1_ceil = math.floor(min_y1), math.ceil(max_y1)
    ori_w = max_x1_ceil - min_x1_floor
    ori_h = max_y1_ceil - min_y1_floor
    ori_x = las.x.astype(float32)
    ori_y = las.y.astype(float32)

    # 存放结果
    new_x = np.zeros_like(ori_x, dtype=np.float32)
    # new_y = np.zeros_like(ori_y, dtype=np.float32)

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])

    for yy_no1 in tqdm(range(len(yy_v_list1[:-1]))):
        # 确定y相关信息
        yy_v1_sta = yy_v_list1[yy_no1]
        yy_v1_end = yy_v_list1[yy_no1 + 1]

        # 确定点云在y上的处理范围
        if yy_no1 == 0 and not (yy_no1 == (len(yy_v_list1[:-1]) - 1)):
            mask_y = ori_y < yy_v1_end
        elif yy_no1 == (len(yy_v_list1[:-1]) - 1):
            mask_y = ori_y >= yy_v1_sta
        else:
            mask_y = (yy_v1_end > ori_y) & (ori_y >= yy_v1_sta)

        inter_ = (yy_v1_end - yy_v1_sta) // 15
        tar_xy_rect = np.meshgrid(np.linspace(min_x1_floor, max_x1_ceil-1, ori_w, dtype=np.float32), np.linspace(min_y1_floor, max_y1_ceil, ori_h, dtype=np.float32))
        # 确定转换矩阵在y上的处理范围
        if yy_no1 == 0 and not (yy_no1 == (len(yy_v_list1[:-1]) - 1)):
            ori_x_rect_reval = xy_rect1[:yy_v1_end-min_y1_floor:inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect1[:yy_v1_end-min_y1_floor:inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][:yy_v1_end-min_y1_floor:inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][:yy_v1_end-min_y1_floor:inter_, ::inter_].ravel().astype(float32)
        elif yy_no1 == (len(yy_v_list1[:-1]) - 1):
            ori_x_rect_reval = xy_rect1[yy_v1_sta-min_y1_floor::inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect1[yy_v1_sta-min_y1_floor::inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][yy_v1_sta-min_y1_floor::inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][yy_v1_sta-min_y1_floor::inter_, ::inter_].ravel().astype(float32)
        else:
            ori_x_rect_reval = xy_rect1[yy_v1_sta - min_y1_floor:yy_v1_end - min_y1_floor:inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect1[yy_v1_sta - min_y1_floor:yy_v1_end - min_y1_floor:inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][yy_v1_sta-min_y1_floor:yy_v1_end-min_y1_floor:inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][yy_v1_sta-min_y1_floor:yy_v1_end-min_y1_floor:inter_, ::inter_].ravel().astype(float32)

        # tar_y_rect_reval = ori_y_rect_reval

        # 计算仿射矩阵
        points = [[ori_x, ori_y, tar_x, tar_y] for ori_x, ori_y, tar_x, tar_y in zip(ori_x_rect_reval, ori_y_rect_reval, tar_x_rect_reval, tar_y_rect_reval)]
        affine_matrix, residuals = compute_affine_matrix(points)
        calculate_residuals(affine_matrix, points)  # 残差计算

        # # 计算x位置
        temp_x = ori_x[mask_y]
        temp_y = ori_y[mask_y]

        tran_x, tran_y = xy_apply_transform(temp_x, temp_y, affine_matrix)
        new_x[mask_y] = tran_x

    las.x = new_x
    new_las = las
    return new_las


def las_reproject1R(las, xy_rect2, xx_range1, yy_range1, xx_range2, yy_range2, yy_step1):
    # 原始信息数据
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x1_floor, max_x1_ceil = math.floor(min_x1), math.ceil(max_x1)
    min_y1_floor, max_y1_ceil = math.floor(min_y1), math.ceil(max_y1)
    ori_w1_int = max_x1_ceil - min_x1_floor
    ori_h1_int = max_y1_ceil - min_y1_floor
    ori_w1 = max_x1 - min_x1
    ori_h1 = max_y1 - min_y1

    min_x2, max_x2 = xx_range2
    min_y2, max_y2 = yy_range2
    min_x2_floor, max_x2_ceil = math.floor(min_x2), math.ceil(max_x2)
    min_y2_floor, max_y2_ceil = math.floor(min_y2), math.ceil(max_y2)
    ori_w2_int = max_x2_ceil - min_x2_floor
    ori_h2_int = max_y2_ceil - min_y2_floor
    ori_w2 = max_x2 - min_x2
    ori_h2 = max_y2 - min_y2

    ori_x = las.x.astype(float32)
    ori_y = las.y.astype(float32)

    scale_x = ori_w1 / ori_w2
    scale_y = ori_h1 / ori_h2

    # 存放结果
    new_x = np.zeros_like(ori_x, dtype=np.float32)
    new_y = np.zeros_like(ori_y, dtype=np.float32)

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])
    yy_v_list2 = (list(np.linspace(min_y2_floor, max_y2_ceil, len(yy_v_list1)).astype(int)))

    for yy_no2 in tqdm(range(len(yy_v_list2[:-1]))):
        # 确定y相关信息
        yy_v1_sta = yy_v_list1[yy_no2]
        yy_v1_end = yy_v_list1[yy_no2 + 1]
        yy_v2_sta = yy_v_list2[yy_no2]
        yy_v2_end = yy_v_list2[yy_no2 + 1]

        # 确定点云在y上的处理范围
        if yy_no2 == 0 and not (yy_no2 == (len(yy_v_list2[:-1]) - 1)):
            mask_y = ori_y < yy_v2_end
        elif yy_no2 == (len(yy_v_list2[:-1]) - 1):
            mask_y = ori_y >= yy_v2_sta
        else:
            mask_y = (yy_v2_end > ori_y) & (ori_y <= yy_v2_sta)

        inter_ = (yy_v2_end - yy_v2_sta) // 15
        tar_xy_rect = np.meshgrid(np.linspace(min_x2_floor, max_x2_ceil - 1, ori_w1_int, dtype=np.float32), np.linspace(min_y2_floor, max_y2_ceil-1, ori_h1_int, dtype=np.float32))
        # 确定转换矩阵在y上的处理范围
        if yy_no2 == 0 and not (yy_no2 == (len(yy_v_list2[:-1]) - 1)):
            ori_x_rect_reval = xy_rect2[:int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect2[:int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][:int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][:int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_].ravel().astype(float32)
        elif yy_no2 == (len(yy_v_list2[:-1]) - 1):
            ori_x_rect_reval = xy_rect2[int((yy_v2_sta - min_y2_floor) * scale_y)::inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect2[int((yy_v2_sta - min_y2_floor) * scale_y)::inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][int((yy_v2_sta - min_y2_floor) * scale_y)::inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][int((yy_v2_sta - min_y2_floor) * scale_y)::inter_, ::inter_].ravel().astype(float32)
        else:
            ori_x_rect_reval = xy_rect2[int((yy_v2_sta - min_y2_floor) * scale_y):int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_, 0].ravel().astype(float32)
            ori_y_rect_reval = xy_rect2[int((yy_v2_sta - min_y2_floor) * scale_y):int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_, 1].ravel().astype(float32)
            tar_x_rect_reval = tar_xy_rect[0][int((yy_v2_sta - min_y2_floor) * scale_y):int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_].ravel().astype(float32)
            tar_y_rect_reval = tar_xy_rect[1][int((yy_v2_sta - min_y2_floor) * scale_y):int((yy_v2_end - min_y2_floor) * scale_y):inter_, ::inter_].ravel().astype(float32)

        # tar_y_rect_reval = ori_y_rect_reval

        # 计算仿射矩阵
        points = [[ori_x, ori_y, tar_x, tar_y] for ori_x, ori_y, tar_x, tar_y in zip(ori_x_rect_reval, ori_y_rect_reval, tar_x_rect_reval, tar_y_rect_reval)]
        affine_matrix, residuals = compute_affine_matrix(points)
        calculate_residuals(affine_matrix, points)  # 残差计算

        # # 计算x位置
        temp_x = ori_x[mask_y].astype(float32)
        temp_y = ori_y[mask_y].astype(float32)

        tran_x, tran_y = xy_apply_transform(temp_x, temp_y, affine_matrix)
        new_x[mask_y] = tran_x

    las.x = ((new_x - min_x2) * scale_x + min_x2).astype(float32)
    las.y = ((ori_y - min_y2) * scale_y + min_y2).astype(float32)
    new_las = las
    return new_las

def las_reproject1(las, xy_rect, xx_range, yy_range, y_off=0):
    # 原始信息数据
    H, W, _ = xy_rect.shape
    min_x, max_x = xx_range
    min_y, max_y = yy_range
    ori_x = np.array(las.x)
    ori_y = np.array(las.y)

    # 投影矩阵处理
    rect_x = xy_rect[:, :, 0]
    max_rect_x, min_rect_x = np.max(rect_x), np.min(rect_x)
    M = int(max_rect_x) - int(min_rect_x)

    xy_mesh = np.meshgrid(np.arange(0, W), np.arange(0, H))
    mesh_x = xy_mesh[0].ravel()
    mesh_y = xy_mesh[1].ravel()

    trans_mesh_x = rect_x[mesh_y, mesh_x]
    max_trans_mesh_x, min_trans_mesh_x = np.max(trans_mesh_x), np.min(trans_mesh_x)

    trans_rect_x = np.zeros([H, M], dtype=np.int32)
    trans_rect_x[:, :int(M / 2)] = int(min_x)
    trans_rect_x[:, int(M / 2):] = int(max_x)
    trans_mesh_x_ = (trans_mesh_x - min_trans_mesh_x).clip(0, M - 1).astype(np.int32)
    trans_rect_x[mesh_y, trans_mesh_x_] = (mesh_x + int(min_x))


    # 点云处理
    index_x = (ori_x - min_trans_mesh_x).clip(0, M - 1).astype(int)
    index_y = (ori_y - min_y).clip(0, H - 1).astype(int)


    # 开始投影
    new_x = trans_rect_x[index_y, index_x]
    new_min_x, new_max_x = np.min(new_x), np.max(new_x)
    scale_x = W / (int(max_x) - int(min_x))
    new_x_ = (new_x - new_min_x) * ((max_x - min_x) * scale_x / (new_max_x - new_min_x)) + min_x

    new_y = ori_y
    # new_min_y, new_max_y = np.min(new_y), np.max(new_y)
    if y_off:
        new_y_ = (new_y - min_y) * ((max_y - min_y + y_off) / (max_y - min_y)) + min_y
    else:
        scale_y = H / (int(max_y) - int(min_y))
        new_y_ = (new_y - min_y) * scale_y + min_y
    las.x = new_x_
    las.y = new_y_

    new_las = las

    return new_las


def las_reproject(las, xy_rect, xx_range, yy_range, y_off=0):
    # ori_x = np.array(las.x - x_min).astype(int)
    # ori_y = np.array(las.y - y_min).astype(int)
    ori_x = np.array(las.x)
    ori_y = np.array(las.y)

    # max_x, min_x = np.max(ori_x), np.min(ori_x)
    # max_y, min_y = np.max(ori_y), np.min(ori_y)
    min_x, max_x = xx_range
    min_y, max_y = yy_range

    h, w, _ = xy_rect.shape

    scale_x = (w) / (int(max_x) - int(min_x))
    scale_y = (h) / (int(max_y) - int(min_y))

    index_x = ((ori_x - min_x) * scale_x).clip(0, w-1).astype(int)
    index_y = ((ori_y - min_y) * scale_y).clip(0, h-1).astype(int)

    # if np.max(ori_x) > w-1 or np.max(ori_y) > h-1:
    #     # pass
    #     ori_x = ori_x.clip(0, w-1)
    #     ori_y = ori_y.clip(0, h-1)

    # new_x = xy_rect[index_y, index_x, 0]
    # new_y = xy_rect[index_y, index_x, 1]

    max_rect_x, min_rect_x = np.max(xy_rect[:, :, 0]), np.min(xy_rect[:, :, 0])
    max_rect_y, min_rect_y = np.max(xy_rect[:, :, 1]), np.min(xy_rect[:, :, 1])

    xy_rect_0 = (xy_rect[:, :, 0] - min_rect_x) * scale_x + min_rect_x
    xy_rect_1 = (xy_rect[:, :, 1] - min_rect_y) * scale_y + min_rect_y
    xy_rect_ = np.concatenate([xy_rect_0[:, :, np.newaxis], xy_rect_1[:, :, np.newaxis]], axis=2)

    new_x = xy_rect_[index_y, index_x, 0]
    new_y = xy_rect_[index_y, index_x, 1]

    new_min_x, new_max_x = np.min(new_x), np.max(new_x)
    new_x_ = (new_x - new_min_x) * ((max_x - min_x) * scale_x / (new_max_x - new_min_x)) + min_x

    if y_off:
        new_min_y, new_max_y = np.min(new_y), np.max(new_y)
        new_y_ = (new_y - new_min_y) * ((new_max_x - new_min_x + y_off) / (new_max_x - new_min_x)) + new_min_y
    else:
        new_y_ = new_y
    las.x = new_x_
    las.y = new_y_

    new_las = las

    return new_las

    # # 创建新文件头
    # new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    #
    # # 写入新文件
    # new_las = laspy.LasData(new_header)
    # new_las.header.offsets = [0, 0, 0]
    # new_las.points.offsets = [0, 0, 0]
    # new_las.header.scales = [0.01, 0.01, 0.001]  # 像素坐标系
    # new_las.points.scales = [0.01, 0.01, 0.001]  # 像素坐标系使用更低精度
    # for dim in las.point_format.dimensions:
    #     new_las[dim.name] = las[dim.name]
    # new_las.xyz = np.array([new_x, new_y, las.z]).astype(np.float64).transpose(1, 0)
    # new_las.x = np.array(new_x).astype(np.float64)
    # new_las.y = np.array(new_y).astype(np.float64)
    # new_las.z = las.z
    #
    # return new_las


def points_Z_zscore_filter(points, z_threshold=3.0):
    """
    基于Z-Score的全局离群值去除（适合快速处理）

    参数：
    - z_max: Z分数阈值（推荐3.0）
    """
    # 计算z坐标轴的Z-Score全局统计量
    mean = np.mean(points.z, axis=0)
    std = np.std(points.z, axis=0)

    # 设置动态阈值
    lower_bound = mean - z_threshold * std
    upper_bound = mean + z_threshold * std

    # 生成高程掩码
    z_valid_mask = (points.z >= lower_bound) & (points.z <= upper_bound)

    # 过滤离群点
    return z_valid_mask


def points_Z_IQR_filter(points, iqr_scale=1.5):
    # 计算z坐标轴的Z-Score全局统计量
    q10, q90 = np.percentile(np.array(points.z).astype(np.float32), [iqr_scale, 100-iqr_scale])
    # iqr = q90 - q10

    # 设置动态阈值
    # lower_bound = q10 - iqr_scale * iqr
    # upper_bound = q90 + iqr_scale * iqr
    lower_bound = q10
    upper_bound = q90

    # 生成高程掩码
    z_valid_mask = (points.z >= lower_bound) & (points.z <= upper_bound)

    # 过滤离群点
    return z_valid_mask


def las_voxel_max_sampling(las, voxel_size=0.5):
    #  ============= las体素格降采样 ================
    # 体素格降采样
    z_values = las.z  # (N, 1)
    xy_values = np.vstack((las.x, las.y)).T  # (N, 2)

    # 计算体素索引
    xy_voxels = np.floor(xy_values / voxel_size).astype(int)

    # 提取Z值并生成排序索引,从大到小
    sorted_indices = np.argsort(z_values)[::-1]

    # 按索引排序xy值数据
    xy_sorted = xy_voxels[sorted_indices]

    # 按体素分组，保留每个体素的第一个点（可改为其他策略）
    _, unique_indices = np.unique(xy_sorted, axis=0, return_index=True)

    # # 最终的降采样点
    # downsample_points = las.points[sorted_indices][unique_indices]

    # 创建新文件并写入筛选后的点
    # las_d = laspy.LasData(las.header)
    # las_d.points = downsample_points
    # 创建新文件并写入筛选后的点
    las_d = SelfLas()
    las_d.x = las.x[sorted_indices][unique_indices]
    las_d.y = las.y[sorted_indices][unique_indices]
    las_d.z = las.z[sorted_indices][unique_indices]

    return las_d



def laspy_voxel_max_sampling(las, voxel_size=0.5):
    #  ============= las体素格降采样 ================
    # 体素格降采样
    z_values = las.z  # (N, 1)
    xy_values = np.vstack((las.x, las.y)).T  # (N, 2)

    # 计算体素索引
    xy_voxels = np.floor(xy_values / voxel_size).astype(int)

    # 提取Z值并生成排序索引,从大到小
    sorted_indices = np.argsort(z_values)[::-1]

    # 按索引排序xy值数据
    xy_sorted = xy_voxels[sorted_indices]

    # 按体素分组，保留每个体素的第一个点（可改为其他策略）
    _, unique_indices = np.unique(xy_sorted, axis=0, return_index=True)

    # # 最终的降采样点
    # downsample_points = las.points[sorted_indices][unique_indices]

    # # 创建新文件并写入筛选后的点
    # # las_d = laspy.LasData(las.header)
    # # las_d.points = downsample_points
    # # 创建新文件并写入筛选后的点
    # las_d = SelfLas()
    # las_d.x = las.x[sorted_indices][unique_indices]
    # las_d.y = las.y[sorted_indices][unique_indices]
    # las_d.z = las.z[sorted_indices][unique_indices]

    return unique_indices


if __name__ == '__main__':
    pass

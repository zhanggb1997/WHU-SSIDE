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

from rasterio import float32
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from disp_rectify import read_points
from points_rectify import compute_affine_matrix, points_apply_transform, calculate_residuals, xy_apply_transform

class SelfLas():
    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

def get_pixel_value(image_path, lon, lat):
    with rasterio.open(image_path) as src:
        x, y = lon, lat

        row, col = src.index(x, y)

        value = src.read(1, window=((row, row + 1), (col, col + 1)))  # 读取第一个波段
        return value[0][0]



def LasInit(las_paths):

    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    all_las = []

    voxel_size = 1

    for las_path in tqdm(las_paths):
        las = laspy.read(las_path)
        las_f = SelfLas()
        las_f.x = np.array(las.x)
        las_f.y = np.array(las.y)
        las_f.z = np.array(las.z)
        all_las.append(las_f)

    merged_points_x = np.concatenate([las.x for las in all_las]).astype(np.float64)
    merged_points_y = np.concatenate([las.y for las in all_las]).astype(np.float64)
    merged_points_z = np.concatenate([las.z for las in all_las]).astype(np.float32)
    merged_las = SelfLas()
    merged_las.x = merged_points_x
    merged_las.y = merged_points_y
    merged_las.z = merged_points_z

    del all_las, merged_points_x, merged_points_y, merged_points_z
    gc.collect()

    return merged_las


def point_localization(las, save_path=None):
    # America
    target_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs("EPSG:6455", target_crs)

    lats, lons = transformer.transform(las.x, las.y)
    path_emg08 = "/home/dshare/06Test/us_nga_egm2008_1.tif"
    eval_dsip = get_pixel_value(path_emg08, np.array(lons)[-1], np.array(lats)[-1])
    #
    heis = 0.3048 * np.array(las.z) + eval_dsip

    # heis = np.array(las.z)

    new_las = SelfLas()
    new_las.x = lons.astype(np.float64)
    new_las.y = lats.astype(np.float64)
    new_las.z = heis.astype(np.float32)

    if save_path:
        point_2_dsm(new_las, save_path, lt_x=np.min(np.array(lons)), lt_y=np.min(np.array(lats)), res_x=0.00001, res_y=-0.00001, crs=target_crs)

    return new_las



def point_projection(las, rpcm):

    x_, y_ = rpcm.projection(las.x, las.y, las.z)

    new_las = SelfLas()

    new_las.x = x_.astype(np.float32)
    new_las.y = y_.astype(np.float32)
    new_las.z = las.z.astype(np.float32)
    return new_las




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

        if res_x != 1 or res_y != 1:
            cols = abs(int((x_max - x_min)/res_x))
            rows = abs(int((y_max - y_min)/res_y))
        else:
            cols = math.ceil(x_max) - math.floor(x_min)
            rows = math.ceil(y_max) - math.floor(y_min)

        dsm, col_edges, row_edges, _ = binned_statistic_2d(
            y, x, values=z,
            statistic='max', bins=(rows, cols),
            # range=[[0, cols], [0, rows]]
        )
        dsm = np.where(np.isnan(dsm), np.nan, dsm)
        transform = from_origin(lt_x, lt_y, res_x, res_y)

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
                    crs=crs,
                    transform=transform,
                    nodata=np.nan
            ) as dst:
                dst.write(dsm, 1)
        except FileExistsError:
            print("dsm.tif")
        dsm_ = dsm
    else:
        dsm_ = None

    return dsm_, x_min, y_min


def point_rectify(las, img, match_split=5, save_path_dsm=None, save_path_img=None, read_path_points=None, save_path_affine_matrix=None):
    dsm_, x_min, y_min = point_2_dsm(las, save_path_dsm)
    img_ = img.get_sub_image([x_min, y_min, dsm_.shape[1], dsm_.shape[0]])
    cv2.imwrite(save_path_img, img_)
    del dsm_, img_
    if os.path.exists(read_path_points):
        points = read_points(read_path_points)
        good_points = []

        for point in points:
            good_points.append([point[0] + x_min, abs(point[1]) + y_min,
                                point[2] + x_min + 0.5, abs(point[3]) + y_min + 0.5])
        affine_matrix, residuals = compute_affine_matrix(good_points)

        calculate_residuals(affine_matrix, good_points)

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

    new_x = np.zeros_like(ori_x, dtype=np.float32)

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])

    for yy_no1 in tqdm(range(len(yy_v_list1[:-1]))):
        yy_v1_sta = yy_v_list1[yy_no1]
        yy_v1_end = yy_v_list1[yy_no1 + 1]

        if yy_no1 == 0 and not (yy_no1 == (len(yy_v_list1[:-1]) - 1)):
            mask_y = ori_y < yy_v1_end
        elif yy_no1 == (len(yy_v_list1[:-1]) - 1):
            mask_y = ori_y >= yy_v1_sta
        else:
            mask_y = (yy_v1_end > ori_y) & (ori_y >= yy_v1_sta)

        inter_ = (yy_v1_end - yy_v1_sta) // 15
        tar_xy_rect = np.meshgrid(np.linspace(min_x1_floor, max_x1_ceil-1, ori_w, dtype=np.float32), np.linspace(min_y1_floor, max_y1_ceil, ori_h, dtype=np.float32))
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


        points = [[ori_x, ori_y, tar_x, tar_y] for ori_x, ori_y, tar_x, tar_y in zip(ori_x_rect_reval, ori_y_rect_reval, tar_x_rect_reval, tar_y_rect_reval)]
        affine_matrix, residuals = compute_affine_matrix(points)
        calculate_residuals(affine_matrix, points)  # 残差计算

        temp_x = ori_x[mask_y]
        temp_y = ori_y[mask_y]

        tran_x, tran_y = xy_apply_transform(temp_x, temp_y, affine_matrix)
        new_x[mask_y] = tran_x

    las.x = new_x
    new_las = las
    return new_las


def las_reproject1R(las, xy_rect2, xx_range1, yy_range1, xx_range2, yy_range2, yy_step1):
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

    new_x = np.zeros_like(ori_x, dtype=np.float32)
    new_y = np.zeros_like(ori_y, dtype=np.float32)

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])
    yy_v_list2 = (list(np.linspace(min_y2_floor, max_y2_ceil, len(yy_v_list1)).astype(int)))

    for yy_no2 in tqdm(range(len(yy_v_list2[:-1]))):
        yy_v1_sta = yy_v_list1[yy_no2]
        yy_v1_end = yy_v_list1[yy_no2 + 1]
        yy_v2_sta = yy_v_list2[yy_no2]
        yy_v2_end = yy_v_list2[yy_no2 + 1]

        if yy_no2 == 0 and not (yy_no2 == (len(yy_v_list2[:-1]) - 1)):
            mask_y = ori_y < yy_v2_end
        elif yy_no2 == (len(yy_v_list2[:-1]) - 1):
            mask_y = ori_y >= yy_v2_sta
        else:
            mask_y = (yy_v2_end > ori_y) & (ori_y <= yy_v2_sta)

        inter_ = (yy_v2_end - yy_v2_sta) // 15
        tar_xy_rect = np.meshgrid(np.linspace(min_x2_floor, max_x2_ceil - 1, ori_w1_int, dtype=np.float32), np.linspace(min_y2_floor, max_y2_ceil-1, ori_h1_int, dtype=np.float32))
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

        points = [[ori_x, ori_y, tar_x, tar_y] for ori_x, ori_y, tar_x, tar_y in zip(ori_x_rect_reval, ori_y_rect_reval, tar_x_rect_reval, tar_y_rect_reval)]
        affine_matrix, residuals = compute_affine_matrix(points)
        calculate_residuals(affine_matrix, points)

        temp_x = ori_x[mask_y].astype(float32)
        temp_y = ori_y[mask_y].astype(float32)

        tran_x, tran_y = xy_apply_transform(temp_x, temp_y, affine_matrix)
        new_x[mask_y] = tran_x

    las.x = ((new_x - min_x2) * scale_x + min_x2).astype(float32)
    las.y = ((ori_y - min_y2) * scale_y + min_y2).astype(float32)
    new_las = las
    return new_las


if __name__ == '__main__':
    pass

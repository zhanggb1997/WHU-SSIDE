'''
Project    : StereoDatasetConstruction
FileName   : get_intersection_region .py
CreateTime : 2025/1/11 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
from copy import deepcopy
import math
import cv2
import numpy as np
import pyproj
import rasterio
import srtm4
from scipy.stats import binned_statistic_2d
from shapely.geometry import Polygon
from pyproj import Transformer
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import gc



def get_epipolar_coeffs_tonggui(xx_range1, yy_range1, yy_step1, rpcm1, rpcm2, alt_levels, alt_eval, RSImg1=None, RSImg2=None):

    epipolar_coeffs1 = []
    epipolar_coeffs2 = []

    y_min, y_max = yy_range1
    x_min, x_max = xx_range1
    y_min_floor, y_max_ceil = math.floor(y_min), math.ceil(y_max)
    x_min_floor, x_max_ceil = math.floor(x_min), math.ceil(x_max)

    yy_v_list = (list(range(y_min_floor, y_max_ceil, yy_step1)) + [y_max_ceil, ])

    for yy_no in tqdm(range(len(yy_v_list[:-1]))):
        yy_v = yy_v_list[yy_no]
        epipolar_coeffs1.append([])
        epipolar_coeffs2.append([])
        yy_center1 = (yy_v + yy_v_list[yy_no+1]) // 2
        for xx_v in range(x_min_floor, x_max_ceil):

            target_col_row1 = []
            for e in alt_levels:
                # print(str(e))
                try:
                    lon_1, lat_1 = rpcm1.localization(int(xx_v), int(yy_center1), e)
                    col_2, row_2 = rpcm2.projection(lon_1, lat_1, e)
                    target_col_row1.append([col_2, row_2])
                except Exception as e:
                    print("x:{}  y:{}  e:{} ".format(str(yy_center1), str(xx_v), e))
            # RSImg2.show_points(np.array(target_col_row1))


            cols1, rows1 = zip(*target_col_row1)
            A1 = np.vstack((np.array(rows1), np.ones(len(rows1)))).T
            m1, c1 = np.linalg.lstsq(A1, cols1, rcond=None)[0]
            epipolar_coeffs1[yy_no].append([m1, -1, c1])
            # print(np.max(np.abs(((m1 * np.array(target_col_row1)[:, 0] + c1)[:, np.newaxis]) - (np.array(target_col_row1)[:, 1][:, np.newaxis]))))


            lon_1, lat_1 = rpcm1.localization(int(xx_v), int(yy_center1), alt_eval)
            col_2_, row_2_ = rpcm2.projection(lon_1, lat_1, alt_eval)

            target_col_row2 = []
            for e in alt_levels:
                lon_2, lat_2 = rpcm2.localization(col_2_, row_2_, e)
                col_1, row_1 = rpcm1.projection(lon_2, lat_2, e)
                target_col_row2.append([col_1, row_1])
            # RSImg1.show_points(np.array(target_col_row2))


            cols2, rows2 = zip(*target_col_row2)
            A2 = np.vstack([rows2, np.ones(len(rows2))]).T
            m2, c2 = np.linalg.lstsq(A2, cols2, rcond=None)[0]
            epipolar_coeffs2[yy_no].append([m2, -1, c2])

    return epipolar_coeffs1, epipolar_coeffs2



def resample_epipolar_imageL_tonggui(img1, epipolar_coeffs2, xx_range1, yy_range1, yy_step1):
    min_y1, max_y1 = yy_range1
    min_x1, max_x1 = xx_range1
    min_y1_floor, max_y1_ceil = math.floor(min_y1), math.ceil(max_y1)
    min_x1_floor, max_x1_ceil = math.floor(min_x1), math.ceil(max_x1)
    img_h1 = max_y1_ceil - min_y1_floor
    img_w1 = max_x1_ceil - min_x1_floor

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])

    y_1a = np.arange(min_y1_floor, max_y1_ceil)
    x_1a = []
    x_1s_offs = []
    x_1s_off = 0

    for yy_no1 in tqdm(range(len(yy_v_list1[:-1]))):
        yy_v1 = yy_v_list1[yy_no1]
        epipolar_coeffs2_y_ = epipolar_coeffs2[yy_no1]

        k1s, _, b1s = np.array(epipolar_coeffs2_y_).transpose(1, 0)
        y_1s = np.arange(yy_v1, yy_v_list1[yy_no1 + 1])
        x_1s_off = (0 if yy_no1 == 0 else ((np.array(epipolar_coeffs2[yy_no1])[:, 0] - np.array(epipolar_coeffs2[yy_no1-1])[:, 0])[np.newaxis, :] * yy_v1 + (np.array(epipolar_coeffs2[yy_no1])[:, 2] - np.array(epipolar_coeffs2[yy_no1-1])[:, 2])[np.newaxis, :]) + x_1s_off)
        x_1s = (((k1s[np.newaxis, :] * y_1s[np.newaxis, :].T + b1s[np.newaxis, :])) - x_1s_off)
        x_1a.append(x_1s)
        x_1s_offs.append(x_1s_off)

    x_1a = (np.concatenate(x_1a, axis=0)).astype(np.float32)

    epipolar_img1 = map_coordinates(
        img1.read(1),
        [y_1a[:, np.newaxis].repeat(img_w1, 1).ravel(), x_1a.ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((img_h1, img_w1))

    xy_a = np.concatenate([x_1a[:, :,np.newaxis], np.arange(min_y1_floor, max_y1_ceil)[:, np.newaxis].repeat(img_w1, 1)[:, :,np.newaxis]], axis=2)

    # 清理内存
    del epipolar_coeffs2, y_1a, x_1a, x_1s_offs
    gc.collect()

    return epipolar_img1, xy_a


def resample_epipolar_imageR_tonggui(img2, epipolar_coeffs1, xx_range1, yy_range1, yy_range2, yy_step1):
    """根据核线方程重采样影像"""
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x1_floor, max_x1_ceil = math.floor(min_x1), math.ceil(max_x1)
    min_y1_floor, max_y1_ceil = math.floor(min_y1), math.ceil(max_y1)
    img_w1 = max_x1_ceil - min_x1_floor
    img_h1 = max_y1_ceil - min_y1_floor

    min_y2, max_y2 = yy_range2
    min_y2_floor, max_y2_ceil = math.floor(min_y2), math.ceil(max_y2)
    img_h2 = max_y2_ceil - min_y2_floor

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])
    yy_v_list2 = (list(np.linspace(min_y2_floor, max_y2_ceil, len(yy_v_list1)).astype(int)))

    # ******  img2  ******
    y_2a = np.linspace(min_y2_floor, max_y2_ceil-1, img_h1).astype(np.float32)
    x_2a = []
    x_2s_offs = []
    x_2s_off = 0

    for yy_no2 in tqdm(range(len((yy_v_list2[:-1])))):
        yy_v2 = yy_v_list2[yy_no2]
        epipolar_coeffs1_y_ = epipolar_coeffs1[yy_no2]

        k2s, _, b2s = np.array(epipolar_coeffs1_y_).transpose(1, 0)
        y_2s = (np.linspace(yy_v2, yy_v_list2[yy_no2 + 1]-1, (yy_v_list1[yy_no2 + 1] - yy_v_list1[yy_no2])))
        x_2s_off = (0 if yy_no2 == 0 else ((np.array(epipolar_coeffs1[yy_no2])[:, 0] - np.array(epipolar_coeffs1[yy_no2-1])[:, 0])[np.newaxis, :] * yy_v2 + (np.array(epipolar_coeffs1[yy_no2])[:, 2] - np.array(epipolar_coeffs1[yy_no2-1])[:, 2])[np.newaxis, :]) + x_2s_off)
        x_2s = (k2s[np.newaxis, :] * y_2s[np.newaxis, :].T + b2s[np.newaxis, :]) - x_2s_off
        x_2a.append(x_2s)
        x_2s_offs.append(x_2s_off)

    x_2a = np.concatenate(x_2a, axis=0).astype(np.float32)

    epipolar_img2 = map_coordinates(
        img2.read(1),
        [(y_2a[:, np.newaxis].repeat(img_w1, 1)).ravel(), x_2a.ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((img_h1, img_w1))

    xy_a = np.concatenate(
        [x_2a[:, :, np.newaxis], y_2a[:, np.newaxis].repeat(img_w1, 1)[:, :, np.newaxis]], axis=2)

    del epipolar_coeffs1, y_2a, x_2a, x_2s_offs
    gc.collect()

    return epipolar_img2, xy_a



def norm_save_epipolar_image(epipolar_img, save_path):
    mean_value = np.nanmean(epipolar_img)
    std_value = np.nanstd(epipolar_img)
    min_value = mean_value - std_value * 2
    max_value = mean_value + std_value * 2
    clip_epipolar_img = np.clip(epipolar_img, min_value, max_value)
    norm_epipolar_img = (255 * ((clip_epipolar_img - min_value) / (max_value - min_value))).astype(np.uint8)

    cv2.imwrite(save_path, norm_epipolar_img)

    return norm_epipolar_img



def get_disp_res_tonggui3(las_pixel1, las_pixel2, yy_range1, xx_range1, yy_range2, xx_range2):

    min_x1, max_x1 = math.floor(xx_range1[0]), math.ceil(xx_range1[1])
    min_y1, max_y1 = math.floor(yy_range1[0]), math.ceil(yy_range1[1])
    min_x2, max_x2 = math.floor(xx_range2[0]), math.ceil(xx_range2[1])
    min_y2, max_y2 = math.floor(yy_range2[0]), math.ceil(yy_range2[1])

    sorted_indices = np.argsort(las_pixel1.z)
    disp_init = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.float32) * -999
    disp_num = np.zeros((max_y1 - min_y1, max_x1 - min_x1), dtype=np.uint8)

    y_ = ((las_pixel1.y - min_y1).astype(np.uint16).clip(0, max_y1 - min_y1 - 1))[sorted_indices]
    x_ = ((las_pixel1.x - min_x1).astype(np.uint16).clip(0, max_x1 - min_x1 - 1))[sorted_indices]
    d = (las_pixel1.y - min_y1) - (las_pixel2.y - min_y2)
    # d = (las_pixel1.y - yy_range1[0]) - (las_pixel2.y - yy_range2[0])
    d_ = d[sorted_indices]

    disp_init[y_, x_] = d_
    return disp_init, disp_num




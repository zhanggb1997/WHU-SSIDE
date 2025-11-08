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

def get_inter(img1, img2):

    # 坐标集合
    lons_lats_1 = [(img1.TLlon, img1.TLlat), (img1.TRlon, img1.TRlat), (img1.BRlon, img1.BRlat), (img1.BLlon, img1.BLlat), (img1.TLlon, img1.TLlat)]
    lons_lats_2 = [(img2.TLlon, img2.TLlat), (img2.TRlon, img2.TRlat), (img2.BRlon, img2.BRlat), (img2.BLlon, img2.BLlat), (img2.TLlon, img2.TLlat)]
    polygon1 = Polygon(lons_lats_1)
    polygon2 = Polygon(lons_lats_2)
    local_coords = polygon1.intersection(polygon2)


    local_coords = list(local_coords.exterior.coords)[:4]

    local_coords_ = [list(inter_bbox) for inter_bbox in local_coords]


    # return re_projected_coords
    return local_coords_


def get_inter_geo(geo_bbox1, geo_bbox2):
    # 使用Shapely构造多边形
    polygon1 = Polygon(geo_bbox1)
    polygon2 = Polygon(geo_bbox2)

    # 计算两多边形的交集
    intersection = polygon1.intersection(polygon2)
    intersection_bbox = list(intersection.exterior.coords)[:4]

    intersection_bbox_ = [list(inter_bbox) for inter_bbox in intersection_bbox]

    return intersection_bbox_

def get_inter_stereo(img1, img2, inter_coords):
    # rpc模型
    rpcm1 = img1.rpc_model
    rpcm2 = img2.rpc_model

    # 依次计算四个角点的高程，并将地理坐标投影到像素坐标系
    inter_pixel_bbox1, inter_pixel_bbox2 = [], []

    # 遍历计算四个坐标点的像素坐标位置
    for lon_, lat_ in inter_coords:
        z_ = srtm4.srtm4(lon_, lat_)
        x1_, y1_ = rpcm1.projection(lon_, lat_, z_)
        x2_, y2_ = rpcm2.projection(lon_, lat_, z_)

        x1_ = max(0, min(x1_, img1.width))
        y1_ = max(0, min(y1_, img1.height))
        x2_ = max(0, min(x2_, img2.width))
        y2_ = max(0, min(y2_, img2.height))

        inter_pixel_bbox1.append([x1_, y1_])
        inter_pixel_bbox2.append([x2_, y2_])

    return inter_pixel_bbox1, inter_pixel_bbox2




def get_split_inter_stereo(inter_geo_bbox, split_times=4):


    if split_times == 1:
        return [inter_geo_bbox]

    split_inter_geo_bboxs = []

    bottom_xs = np.linspace(min(np.array(inter_geo_bbox)[:, 0]), max(np.array(inter_geo_bbox)[:, 0]), split_times + 1)
    top_xs = np.linspace(min(np.array(inter_geo_bbox)[:, 0]), max(np.array(inter_geo_bbox)[:, 0]), split_times + 1)
    bottom_ys = np.linspace(min(np.array(inter_geo_bbox)[:, 1]), min(np.array(inter_geo_bbox)[:, 1]), split_times + 1)
    top_ys = np.linspace(max(np.array(inter_geo_bbox)[:, 1]), max(np.array(inter_geo_bbox)[:, 1]), split_times + 1)

    for i in range(split_times):
        new_inter_geo_bbox = deepcopy(inter_geo_bbox)
        new_inter_geo_bbox[0][0] = bottom_xs[i]
        new_inter_geo_bbox[3][0] = bottom_xs[i+1]
        new_inter_geo_bbox[1][0] = top_xs[i]
        new_inter_geo_bbox[2][0] = top_xs[i+1]
        new_inter_geo_bbox[0][1] = bottom_ys[i]
        new_inter_geo_bbox[3][1] = bottom_ys[i+1]
        new_inter_geo_bbox[1][1] = top_ys[i]
        new_inter_geo_bbox[2][1] = top_ys[i+1]

        split_inter_geo_bboxs.append(new_inter_geo_bbox)

    return split_inter_geo_bboxs


def get_center_lon(inter_coords):
    lon_sum = 0
    for lon_, lat_ in inter_coords[:4]:
        lon_sum += lon_
    lon_eval = (lon_sum / 4)

    return lon_eval

def get_center_x(inter_coords):
    xx_sum = 0
    for xx_, yy_ in inter_coords[:4]:
        xx_sum += xx_
    xx_eval = (xx_sum / 4)

    return xx_eval

def get_center_y(inter_coords):
    yy_sum = 0
    for xx_, yy_ in inter_coords[:4]:
        yy_sum += yy_
    yy_eval = (yy_sum / 4)

    return yy_eval


def get_altitude_range(coords):
    lon_m = min(np.array(coords)[:, 0])
    lon_M = max(np.array(coords)[:, 0])
    lat_m = min(np.array(coords)[:, 1])
    lat_M = max(np.array(coords)[:, 1])

    # s = 0.001 / 12  # SRTM90 pixel spacing is 0.001 / 12 degrees
    s = 0.001  # SRTM90 pixel spacing is 0.001 / 12 degrees
    points = [(lon, lat) for lon in np.arange(lon_m, lon_M, s)
                         for lat in np.arange(lat_m, lat_M, s)]
    lons, lats = np.asarray(points).T
    alts = srtm4.srtm4(lons, lats)  # TODO use srtm4 nn interpolation option
    h_m = min(alts)
    h_M = max(alts)
    h_e = np.mean(alts)

    return h_m, h_M, h_e



def get_epipolar_coeffs_tonggui(xx_range1, yy_range1, yy_step1, rpcm1, rpcm2, alt_levels, alt_eval, RSImg1=None, RSImg2=None):
    # 对每个像素生成投影轨迹
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
            # 在当前像素位置下遍历高程获取核线曲线
            target_col_row1 = []
            for e in alt_levels:
                # print(str(e))
                try:
                    lon_1, lat_1 = rpcm1.localization(int(xx_v), int(yy_center1), e)
                    col_2, row_2 = rpcm2.projection(lon_1, lat_1, e)
                    target_col_row1.append([col_2, row_2])
                except Exception as e:
                    print("x:{}  y:{}  e:{}  RPCM 跳过！".format(str(yy_center1), str(xx_v), e))
            # RSImg2.show_points(np.array(target_col_row1))

            # 在右视上直线拟合核线方程：ax + by + c = 0
            cols1, rows1 = zip(*target_col_row1)
            A1 = np.vstack((np.array(rows1), np.ones(len(rows1)))).T
            m1, c1 = np.linalg.lstsq(A1, cols1, rcond=None)[0]  # 求解线性最小二乘问题
            epipolar_coeffs1[yy_no].append([m1, -1, c1])  # 转换为标准直线方程
            # print(np.max(np.abs(((m1 * np.array(target_col_row1)[:, 0] + c1)[:, np.newaxis]) - (np.array(target_col_row1)[:, 1][:, np.newaxis]))))

            # 获取当前核线中心像素+平均高度下获取的另一端像点位置
            lon_1, lat_1 = rpcm1.localization(int(xx_v), int(yy_center1), alt_eval)
            col_2_, row_2_ = rpcm2.projection(lon_1, lat_1, alt_eval)
            # 在当前像素位置下遍历高程获取核线曲线
            target_col_row2 = []
            for e in alt_levels:
                lon_2, lat_2 = rpcm2.localization(col_2_, row_2_, e)
                col_1, row_1 = rpcm1.projection(lon_2, lat_2, e)
                target_col_row2.append([col_1, row_1])
            # RSImg1.show_points(np.array(target_col_row2))

            # 在右视上直线拟合核线方程：ax + by + c = 0
            cols2, rows2 = zip(*target_col_row2)
            A2 = np.vstack([rows2, np.ones(len(rows2))]).T
            m2, c2 = np.linalg.lstsq(A2, cols2, rcond=None)[0]  # 求解线性最小二乘问题
            epipolar_coeffs2[yy_no].append([m2, -1, c2])  # 转换为标准直线方程
            # print(np.max(np.abs(((m2 * np.array(target_col_row2)[:,0] + c2)[:, np.newaxis]) - (np.array(target_col_row2)[:,1][:, np.newaxis]))))

            # RSImg1.draw_lines(epipolar_coeffs2[xx_no][-1], target_col_row2, 1024, "./temp_epi/epi_L_{}.png".format(str(yy_v)))
            # RSImg2.draw_lines(epipolar_coeffs1[xx_no][-1], target_col_row1, 512, "./temp_epi/epi_R_{}.png".format(str(yy_v)))

    return epipolar_coeffs1, epipolar_coeffs2


def resample_epipolar_images(img1, img2, epipolar_coeffs1, epipolar_coeffs2, yy_range1, xx_range1, xx_range2, xx_step1):
    """根据核线方程重采样影像"""
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x2, max_x2 = xx_range2
    xx_v_list1 = (list(range(xx_range1[0], xx_range1[1], xx_step1)) + [max_x1, ])
    xx_v_list2 = (list(range(xx_range2[0], xx_range2[1], int(xx_step1 * ((max_x2 - min_x2) / (max_x1 - min_x1))))) + [max_x2, ])

    # ******  img1  ******
    x_1a = np.arange(min_x1, max_x1).astype(np.float32)
    y_1a = []
    y_1s_offs = []
    y_1s_off = 0

    for xx_no1 in tqdm(range(len(xx_v_list1[:-1]))):
        xx_v1 = xx_v_list1[xx_no1]
        epipolar_coeffs2_x_ = epipolar_coeffs2[xx_no1]

        k1s, _, b1s = np.array(epipolar_coeffs2_x_).transpose(1, 0)
        x_1s = np.arange(xx_v1, xx_v_list1[xx_no1 + 1])
        y_1s = ((k1s[:, np.newaxis] * x_1s[:, np.newaxis].T + b1s[:, np.newaxis]))
        # y_1s_off = 0
        # y_1s_off = (0 if xx_no1 == 0 else ((np.array(epipolar_coeffs2[xx_no1])[:, 0] - np.array(epipolar_coeffs2[xx_no1-1])[:, 0])[:, np.newaxis] * xx_v1 + (np.array(epipolar_coeffs2[xx_no1])[:, 2] - np.array(epipolar_coeffs2[xx_no1-1])[:, 2])[:, np.newaxis]) + y_1s_off)
        # y_1s = (((k1s[:, np.newaxis] * x_1s[:, np.newaxis].T + b1s[:, np.newaxis])) - y_1s_off)
        y_1a.append(y_1s)
        y_1s_offs.append(y_1s_off)

    y_1a = (np.concatenate(y_1a, axis=1)).astype(np.float32)

    epipolar_img1 = map_coordinates(
        img1.read(1),
        [y_1a.ravel(), x_1a[np.newaxis, :].repeat(max_y1-min_y1, 0).ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((max_y1 - min_y1, max_x1 - min_x1))

    epi_dict = {(i+min_x1, int(value)): j for (j, i), value in np.ndenumerate(y_1a)}

    # 清理内存
    del epipolar_coeffs2, x_1a, y_1a, y_1s_offs
    gc.collect()


    # ******  img2  ******
    x_2a = np.linspace(min_x2, max_x2, (max_x1 - min_x1)).astype(np.float32)
    y_2a = []
    y_2s_offs = []
    y_2s_off = 0

    for xx_no2 in tqdm(range(len((xx_v_list2[:-1])))):
        xx_v2 = xx_v_list2[xx_no2]
        epipolar_coeffs1_x_ = epipolar_coeffs1[xx_no2]

        k2s, _, b2s = np.array(epipolar_coeffs1_x_).transpose(1, 0)
        x_2s = (np.linspace(xx_v2, xx_v_list2[xx_no2 + 1], (xx_v_list1[xx_no2 + 1] - xx_v_list1[xx_no2])))
        # y_2s = (k2s[:, np.newaxis] * x_2s[:, np.newaxis].T + b2s[:, np.newaxis])
        y_2s_off = (0 if xx_no2 == 0 else ((np.array(epipolar_coeffs1[xx_no2])[:, 0] - np.array(epipolar_coeffs1[xx_no2-1])[:, 0])[:, np.newaxis] * xx_v2 + (np.array(epipolar_coeffs1[xx_no2])[:, 2] - np.array(epipolar_coeffs1[xx_no2-1])[:, 2])[:, np.newaxis]) + y_2s_off)
        # y_2s_off = 0
        y_2s = (k2s[:, np.newaxis] * x_2s[:, np.newaxis].T + b2s[:, np.newaxis]) - y_2s_off
        y_2a.append(y_2s)
        y_2s_offs.append(y_2s_off)

    y_2a = np.concatenate(y_2a, axis=1).astype(np.float32)

    epipolar_img2 = map_coordinates(
        img2.read(1),
        [y_2a.ravel(), x_2a[np.newaxis, :].repeat(max_y1-min_y1, 0).ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((max_y1 - min_y1, max_x1 - min_x1))

    # 清理内存
    del epipolar_coeffs1, x_2a, y_2a, y_2s_offs
    gc.collect()

    return epipolar_img1, epipolar_img2, epi_dict



def resample_epipolar_imageL(img1, epipolar_coeffs2, yy_range1, xx_range1, xx_step1):
    """根据核线方程重采样影像"""
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    xx_v_list1 = (list(range(xx_range1[0], xx_range1[1], xx_step1)) + [max_x1, ])

    # ******  img1  ******
    x_1a = np.arange(min_x1, max_x1).astype(np.float32)
    y_1a = []
    y_1s_offs = []
    y_1s_off = 0

    for xx_no1 in tqdm(range(len(xx_v_list1[:-1]))):
        xx_v1 = xx_v_list1[xx_no1]
        epipolar_coeffs2_x_ = epipolar_coeffs2[xx_no1]

        k1s, _, b1s = np.array(epipolar_coeffs2_x_).transpose(1, 0)
        x_1s = np.arange(xx_v1, xx_v_list1[xx_no1 + 1])
        # y_1s = ((k1s[:, np.newaxis] * x_1s[:, np.newaxis].T + b1s[:, np.newaxis]))
        y_1s_off = (0 if xx_no1 == 0 else ((np.array(epipolar_coeffs2[xx_no1])[:, 0] - np.array(epipolar_coeffs2[xx_no1-1])[:, 0])[:, np.newaxis] * xx_v1 + (np.array(epipolar_coeffs2[xx_no1])[:, 2] - np.array(epipolar_coeffs2[xx_no1-1])[:, 2])[:, np.newaxis]) + y_1s_off)
        y_1s = (((k1s[:, np.newaxis] * x_1s[:, np.newaxis].T + b1s[:, np.newaxis])) - y_1s_off)
        y_1a.append(y_1s)
        y_1s_offs.append(y_1s_off)

    y_1a = (np.concatenate(y_1a, axis=1)).astype(np.float32)

    epipolar_img1 = map_coordinates(
        img1.read(1),
        [y_1a.ravel(), x_1a[np.newaxis, :].repeat(max_y1-min_y1, 0).ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((max_y1 - min_y1, max_x1 - min_x1))

    epi_dict = {(i+min_x1, int(value)): j for (j, i), value in np.ndenumerate(y_1a)}

    # 清理内存
    del epipolar_coeffs2, x_1a, y_1a, y_1s_offs
    gc.collect()

    return epipolar_img1, epi_dict


def resample_epipolar_imageR(img2, epipolar_coeffs1, yy_range1, xx_range1, xx_range2, xx_step1):
    """根据核线方程重采样影像"""
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x2, max_x2 = xx_range2
    xx_v_list1 = (list(range(xx_range1[0], xx_range1[1], xx_step1)) + [max_x1, ])
    xx_v_list2 = (list(range(xx_range2[0], xx_range2[1], int(xx_step1 * ((max_x2 - min_x2) / (max_x1 - min_x1))))) + [max_x2, ])

    # ******  img2  ******
    x_2a = np.linspace(min_x2, max_x2, (max_x1 - min_x1)).astype(np.float32)
    y_2a = []
    y_2s_offs = []
    y_2s_off = 0

    for xx_no2 in tqdm(range(len((xx_v_list2[:-1])))):
        xx_v2 = xx_v_list2[xx_no2]
        epipolar_coeffs1_x_ = epipolar_coeffs1[xx_no2]

        k2s, _, b2s = np.array(epipolar_coeffs1_x_).transpose(1, 0)
        x_2s = (np.linspace(xx_v2, xx_v_list2[xx_no2 + 1], (xx_v_list1[xx_no2 + 1] - xx_v_list1[xx_no2])))
        # y_2s = (k2s[:, np.newaxis] * x_2s[:, np.newaxis].T + b2s[:, np.newaxis])
        y_2s_off = (0 if xx_no2 == 0 else ((np.array(epipolar_coeffs1[xx_no2])[:, 0] - np.array(epipolar_coeffs1[xx_no2-1])[:, 0])[:, np.newaxis] * xx_v2 + (np.array(epipolar_coeffs1[xx_no2])[:, 2] - np.array(epipolar_coeffs1[xx_no2-1])[:, 2])[:, np.newaxis]) + y_2s_off)
        y_2s = (k2s[:, np.newaxis] * x_2s[:, np.newaxis].T + b2s[:, np.newaxis]) - y_2s_off
        y_2a.append(y_2s)
        y_2s_offs.append(y_2s_off)

    y_2a = np.concatenate(y_2a, axis=1).astype(np.float32)

    epipolar_img2 = map_coordinates(
        img2.read(1),
        [y_2a.ravel(), x_2a[np.newaxis, :].repeat(max_y1-min_y1, 0).ravel()],
        order=1,
        mode='constant',
        cval=0
    ).reshape((max_y1 - min_y1, max_x1 - min_x1))

    # 清理内存
    del epipolar_coeffs1, x_2a, y_2a, y_2s_offs
    gc.collect()

    return epipolar_img2


def resample_epipolar_imageL_tonggui(img1, epipolar_coeffs2, xx_range1, yy_range1, yy_step1):
    """根据核线方程重采样影像"""
    min_y1, max_y1 = yy_range1
    min_x1, max_x1 = xx_range1
    min_y1_floor, max_y1_ceil = math.floor(min_y1), math.ceil(max_y1)
    min_x1_floor, max_x1_ceil = math.floor(min_x1), math.ceil(max_x1)
    img_h1 = max_y1_ceil - min_y1_floor
    img_w1 = max_x1_ceil - min_x1_floor

    yy_v_list1 = (list(range(min_y1_floor, max_y1_ceil, yy_step1)) + [max_y1_ceil, ])

    # ******  img1  ******
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
    # return epipolar_img1, epi_dict1, xy_a


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

    # 清理内存
    del epipolar_coeffs1, y_2a, x_2a, x_2s_offs
    gc.collect()

    return epipolar_img2, xy_a



def norm_save_epipolar_image(epipolar_img, save_path):
    # # # ========== std mean 方式
    mean_value = np.nanmean(epipolar_img)
    std_value = np.nanstd(epipolar_img)
    min_value = mean_value - std_value * 2
    max_value = mean_value + std_value * 2
    clip_epipolar_img = np.clip(epipolar_img, min_value, max_value)
    norm_epipolar_img = (255 * ((clip_epipolar_img - min_value) / (max_value - min_value))).astype(np.uint8)
    # # # ========== 2% 98% 方式
    # # 计算2%和98%分位数
    # min_value, max_value = np.percentile(epipolar_img, (1, 99))
    # clip_epipolar_img = np.clip(epipolar_img, min_value, max_value)
    # norm_epipolar_img = (255 * ((clip_epipolar_img - min_value) / (max_value - min_value))).astype(np.uint8)

    cv2.imwrite(save_path, norm_epipolar_img)

    return norm_epipolar_img



def get_disp_res(las_pixel1, las_pixel2, bbox1, bbox2, epi_dict):
    min_x1, max_x1 = int(np.min(np.array(bbox1)[:, 0])), int(np.max(np.array(bbox1)[:, 0])) + 1
    min_y1, max_y1 = int(np.min(np.array(bbox1)[:, 1])) + 1, int(np.max(np.array(bbox1)[:, 1])) - 1
    min_x2, max_x2 = int(np.min(np.array(bbox2)[:, 0])), int(np.max(np.array(bbox2)[:, 0])) + 1
    min_y2, max_y2 = int(np.min(np.array(bbox2)[:, 1])) + 1, int(np.max(np.array(bbox2)[:, 1])) - 1

    disparity = (las_pixel1.x - min_x1) - ((las_pixel2.x - min_x2) * (max_x1-min_x1)/(max_x2-min_x2))

    # x_ = np.round(las_pixel1.x - min_x1).astype(np.uint16)
    x_ = (las_pixel1.x - min_x1).astype(np.uint16)

    disp_init = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.float32) * -999

    disp_num = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.uint8)

    for i, xyz in enumerate(las_pixel1.xyz):
        if (int(xyz[0]), int(xyz[1])) in epi_dict.keys():
            epi_y = epi_dict[(int(xyz[0]), int(xyz[1]))]
            if disp_init[epi_y, x_[i]] != -999 and abs(disp_init[epi_y, x_[i]]) < abs(disparity[i]):
                disp_init[epi_y, x_[i]] = disparity[i]
            # if disp_init[epi_y, x_[i]] > disparity[i]:
            # if disp_init[epi_y, x_[i]] != -999:
            #     disp_init[epi_y, x_[i]] += disparity[i]
                # disp_num[epi_y, x_[i]] += 1
            else:
                disp_init[epi_y, x_[i]] = disparity[i]
        # else:
        #     print("缺失：" + str(int(xyz[0])) + "," + str(int(xyz[1])))

    # # disp_num[disp_num==0] = 1
    # disp_res = disp_init / disp_num

    # return disp_res
    return disp_init






def get_disp_res_tonggui(las_pixel1, las_pixel2, yy_range1, xx_range1, yy_range2, xx_range2, epi_dict):
    # min_x1, max_x1 = xx_range1
    # min_y1, max_y1 = yy_range1
    # min_x2, max_x2 = xx_range2
    # min_y2, max_y2 = yy_range2

    # m_ = []
    # x_ = []
    # for xyz in las_pixel1.xyz:
    #     if (int(xyz[1]), int(xyz[0])) in epi_dict.keys():
    #         x_.append(epi_dict[(int(xyz[1]), int(xyz[0]))])
    #         m_.append(True)
    #     else:
    #         # print("缺失：" + str(int(xyz[1])) + "," + str(int(xyz[0])))
    #         m_.append(False)
    # d_ = ((las_pixel1.y - int(min_y1)) - ((las_pixel2.y - int(min_y2)) * (int(max_y1) - int(min_y1)) / (int(max_y2) - int(min_y2))))[m_]
    # y_ = (las_pixel1.y - int(min_y1))[m_]
    #
    # # 创建栅格网格
    # cols = int(int(max_x1) - int(min_x1))
    # rows = int(int(max_y1) - int(min_y1))
    #
    # # 使用分箱统计计算每个像素的最大高程
    # disp, col_edges, row_edges, _ = binned_statistic_2d(
    #     y_, x_, values=d_,
    #     statistic='max', bins=(rows, cols),
    #     # range=[[0, cols], [0, rows]]
    # )
    #
    # # 处理缺失值（无点的像素设为NaN）
    # disp_init = np.where(np.isnan(disp), -999, disp)

    min_x1, max_x1 = int(xx_range1[0]), int(xx_range1[1])
    min_y1, max_y1 = int(yy_range1[0]), int(yy_range1[1])
    min_x2, max_x2 = int(xx_range2[0]), int(xx_range2[1])
    min_y2, max_y2 = int(yy_range2[0]), int(yy_range2[1])

    disparity = (las_pixel1.y - min_y1) - ((las_pixel2.y - min_y2) * (max_y1-min_y1)/(max_y2-min_y2))
    y_ = (las_pixel1.y - min_y1).astype(np.uint16)
    disp_init = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.float32) * -999
    disp_num = np.zeros((max_y1 - min_y1, max_x1 - min_x1), dtype=np.uint8)

    for i, xyz in enumerate(las_pixel1.xyz):
        if xyz[0] < 0:
            continue
        if xyz[1] < 0:
            continue
        if (int(xyz[1]), int(xyz[0])) in epi_dict.keys():
            epi_x = epi_dict[(int(xyz[1]), int(xyz[0]))]
            # if disp_init[y_[i], epi_x] != -999 and abs(disp_init[y_[i], epi_x]) < abs(disparity[i]):
            if disp_init[y_[i], epi_x] < disparity[i]:
                disp_init[y_[i], epi_x] = disparity[i]
            disp_num[y_[i], epi_x] += 1
        # else:
        #     print("缺失：" + str(int(xyz[1])) + "," + str(int(xyz[0])))

    # # disp_num[disp_num==0] = 1
    # disp_res = disp_init / disp_num
    #
    # # return disp_res
    return disp_init, disp_num




def get_disp_res_tonggui2(las_pixel1, las_pixel2, yy_range1, xx_range1, yy_range2, xx_range2):

    min_x1, max_x1 = int(xx_range1[0]), int(xx_range1[1])
    min_y1, max_y1 = int(yy_range1[0]), int(yy_range1[1])
    min_x2, max_x2 = int(xx_range2[0]), int(xx_range2[1])
    min_y2, max_y2 = int(yy_range2[0]), int(yy_range2[1])

    # disparity = (las_pixel1.y - min_y1) - ((las_pixel2.y - min_y2) * (max_y1-min_y1)/(max_y2-min_y2))
    disparity = (las_pixel1.y - min_y1) - (las_pixel2.y - min_y2)
    y_ = (las_pixel1.y - min_y1).astype(np.uint16).clip(0, max_y1 - min_y1 - 1)
    x_ = (las_pixel1.x - min_x1).astype(np.uint16).clip(0, max_x1 - min_x1 - 1)
    disp_init = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.float32) * -999
    disp_num = np.zeros((max_y1 - min_y1, max_x1 - min_x1), dtype=np.uint8)

    for i in range(len(las_pixel1.xyz)):
        if disp_init[y_[i], x_[i]] < disparity[i]:
            disp_init[y_[i], x_[i]] = disparity[i]
            disp_num[y_[i], x_[i]] += 1
        # else:
        #     print("缺失：" + str(int(xyz[1])) + "," + str(int(xyz[0])))

    # # disp_num[disp_num==0] = 1
    # disp_res = disp_init / disp_num
    #
    # # return disp_res
    return disp_init, disp_num

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




def get_disp_res_epi_tonggui(las_pixel1, las_pixel2, yy_range1, xx_range1, yy_range2, xx_range2, epi_dict):
    min_x1, max_x1 = xx_range1
    min_y1, max_y1 = yy_range1
    min_x2, max_x2 = xx_range2
    min_y2, max_y2 = yy_range2

    m_ = []
    x_ = []
    for xyz in las_pixel1.xyz:
        if (int(xyz[1]), int(xyz[0])) in epi_dict.keys():
            x_.append(epi_dict[(int(xyz[1]), int(xyz[0]))])
            m_.append(True)
        else:
            # print("缺失：" + str(int(xyz[1])) + "," + str(int(xyz[0])))
            m_.append(False)
    d_ = ((las_pixel1.y - int(min_y1)) - ((las_pixel2.y - int(min_y2)) * (int(max_y1) - int(min_y1)) / (int(max_y2) - int(min_y2))))[m_]
    y_ = (las_pixel1.y - int(min_y1))[m_]

    # 创建栅格网格
    cols = int(int(max_x1) - int(min_x1)) + 1
    rows = int(int(max_y1) - int(min_y1)) + 1

    # 使用分箱统计计算每个像素的最大高程
    disp, col_edges, row_edges, _ = binned_statistic_2d(
        y_, x_, values=d_,
        statistic='max', bins=(rows, cols),
        # range=[[0, cols], [0, rows]]
    )

    # 处理缺失值（无点的像素设为NaN）
    disp_init = np.where(np.isnan(disp), -999, disp)



    # disparity = (las_pixel1.y - min_y1) - ((las_pixel2.y - min_y2) * (max_y1-min_y1)/(max_y2-min_y2))
    # y_ = (las_pixel1.y - min_y1).astype(np.uint16)
    # disp_init = np.ones((max_y1 - min_y1, max_x1 - min_x1), dtype=np.float32) * -999
    # for i, xyz in enumerate(las_pixel1.xyz):
    #     if xyz[0] < 0:
    #         continue
    #     if xyz[1] < 0:
    #         continue
    #     if (int(xyz[1]), int(xyz[0])) in epi_dict.keys():
    #         epi_x = epi_dict[(int(xyz[1]), int(xyz[0]))]
    #         # if disp_init[y_[i], epi_x] != -999 and abs(disp_init[y_[i], epi_x]) < abs(disparity[i]):
    #         if disp_init[y_[i], epi_x] != -999 and disp_init[y_[i], epi_x] < disparity[i]:
    #             disp_init[y_[i], epi_x] = disparity[i]
    #         else:
    #             disp_init[y_[i], epi_x] = disparity[i]
    #     # else:
    #     #     print("缺失：" + str(int(xyz[1])) + "," + str(int(xyz[0])))
    #
    # # # disp_num[disp_num==0] = 1
    # # disp_res = disp_init / disp_num
    #
    # # return disp_res
    return disp_init



def get_points_cal_off(sift_path, yy_range1, xx_range1, yy_range2, xx_range2):
    points = []
    points_1 = []
    points_2 = []

    with open(sift_path, 'r') as file:
        for line in file:
            # 去除行末换行符并按空格分割字符串
            row = line.strip().split('\t')
            # 将字符串转换为浮点数（或整数）
            row = [float(num) for num in row]
            points.append(row)

    for pts in points:
        x_1, y_1, x_2, y_2 = pts  # 关键点位置 x,y 像素坐标
        # if xx_range1[0]<=x_1<xx_range1[1] & yy_range1[0]<=y_1<yy_range1[1] & xx_range2[0]<=x_2<xx_range2[1] & yy_range2[0]<=y_2<yy_range2[1]:
        if xx_range1[0]<=x_1<xx_range1[1] & yy_range1[0]<=y_1<yy_range1[1]:
            points_1.append([x_1, y_1])
            points_2.append([x_2, y_2])

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffine2D(np.array(points_2), np.array(points_1))

    # 提取x方向和y方向的变化斜率
    slope_x_vs_xb = M[0, 0]  # a1
    slope_x_vs_yb = M[0, 1]  # a2
    slope_y_vs_xb = M[1, 0]  # b1
    slope_y_vs_yb = M[1, 1]  # b2

    print(f"x方向相对于影像B的x坐标的斜率: {slope_x_vs_xb:.4f}")
    print(f"x方向相对于影像B的y坐标的斜率: {slope_x_vs_yb:.4f}")
    print(f"y方向相对于影像B的x坐标的斜率: {slope_y_vs_xb:.4f}")
    print(f"y方向相对于影像B的y坐标的斜率: {slope_y_vs_yb:.4f}")



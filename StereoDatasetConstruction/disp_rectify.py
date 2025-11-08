'''
Project    : RSDeploy
FileName   : disp_rectify .py
CreateTime : 2025/2/26 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import cv2
import numpy as np
from scipy.ndimage import map_coordinates



def read_points(file_path):
    """
    从txt文件读取同名点对，格式：xdsm[tab]ydsm[tab]ximg[tab]yimg
    """
    points = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            parts = line.split('\t')
            if len(parts) != 4:
                print(f"警告：第{line_num}行格式错误，已跳过")
                continue
            try:
                x_dsm = float(parts[0])
                y_dsm = abs(float(parts[1]))
                x_img = float(parts[2])
                y_img = abs(float(parts[3]))
                points.append((x_dsm, y_dsm, x_img, y_img))
            except ValueError:
                print(f"错误：第{line_num}行包含非数值数据，已跳过")
    return points



def img_apply_transform(ori_img, matrix):
    """对读取的影像应用仿射变换测试点"""
    # 原始影像处理
    height, width = ori_img.shape[:2]

    y_mgrid, x_mgrid = np.mgrid[0:height, 0:width]

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_mgrid)
    coords = np.stack([x_mgrid, y_mgrid, ones], axis=0)

    # 计算逆/变换矩阵
    # inv_matrix = np.linalg.inv(matrix)
    inv_matrix = matrix

    # 找到原始对应坐标
    src_coords = np.tensordot(inv_matrix, coords, axes=1)  # 形状 (3, height, width)
    src_x, src_y = src_coords[0], src_coords[1]  # 提取x和y坐标

    # 使用双线性插值从原始DSM中获取值
    transformed_dsm = map_coordinates(ori_img, [src_y, src_x], order=0, mode='constant', cval=-999)
    transformed_dsm[transformed_dsm == 0] = -999

    # transformed_dsm_ = np.ones_like(transformed_dsm) * -999
    # if x_ < 0:
    #     transformed_dsm_[:, :int(x_)] = transformed_dsm[:, -int(x_):]

    return transformed_dsm

def img1_apply_transform(ori_img, matrix):
    """对读取的影像应用仿射变换测试点"""
    # 原始影像处理
    height, width = ori_img.shape[:2]

    y_mgrid, x_mgrid = np.mgrid[0:height, 0:width]

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_mgrid)
    coords = np.stack([x_mgrid, y_mgrid, ones], axis=0)

    # 计算逆/变换矩阵
    # inv_matrix = np.linalg.inv(matrix)
    inv_matrix = matrix

    # 找到原始对应坐标
    src_coords = np.tensordot(inv_matrix, coords, axes=1)  # 形状 (3, height, width)
    src_x, src_y = src_coords[0], src_coords[1]  # 提取x和y坐标

    # # 计算仿射矩阵
    # step_h, step_w = height // 16, width // 16
    # points = [[ori_x, ori_y, tar_x, tar_y] for ori_x, ori_y, tar_x, tar_y in
    #           zip(src_x[::step_h, ::step_w].ravel().astype(np.float32), src_y[::step_h, ::step_w].ravel().astype(np.float32),
    #               x_mgrid[::step_h, ::step_w].ravel().astype(np.float32), y_mgrid[::step_h, ::step_w].ravel().astype(np.float32))]
    # affine_matrix, residuals = compute_affine_matrix(points)
    # from StereoDatasetConstruction.points_rectify import calculate_residuals
    # calculate_residuals(affine_matrix, points)  # 残差计算

    # 使用双线性插值从原始DSM中获取值
    transformed_dsm = map_coordinates(ori_img, [src_y, src_x], order=1, mode='constant', cval=-999)
    transformed_dsm[transformed_dsm == 0] = -999

    # transformed_dsm_ = np.ones_like(transformed_dsm) * -999
    # if x_ < 0:
    #     transformed_dsm_[:, :int(x_)] = transformed_dsm[:, -int(x_):]

    return transformed_dsm

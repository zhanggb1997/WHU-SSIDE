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
    points = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                continue
            try:
                x_dsm = float(parts[0])
                y_dsm = abs(float(parts[1]))
                x_img = float(parts[2])
                y_img = abs(float(parts[3]))
                points.append((x_dsm, y_dsm, x_img, y_img))
            except ValueError:
                print(f"error")
    return points




def img1_apply_transform(ori_img, matrix):
    height, width = ori_img.shape[:2]

    y_mgrid, x_mgrid = np.mgrid[0:height, 0:width]

    ones = np.ones_like(x_mgrid)
    coords = np.stack([x_mgrid, y_mgrid, ones], axis=0)

    inv_matrix = matrix

    src_coords = np.tensordot(inv_matrix, coords, axes=1)  # 形状 (3, height, width)
    src_x, src_y = src_coords[0], src_coords[1]  # 提取x和y坐标

    transformed_dsm = map_coordinates(ori_img, [src_y, src_x], order=1, mode='constant', cval=-999)
    transformed_dsm[transformed_dsm == 0] = -999


    return transformed_dsm

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


def compute_affine_matrix(points):
    """
    使用最小二乘法计算仿射变换矩阵
    """
    if len(points) < 3:
        raise ValueError("至少需要3个非共线点以计算仿射变换")

    # 构建最小二乘方程组 A * theta = b
    A = []
    b = []
    for x_dsm, y_dsm, x_img, y_img in points:
        # x' 方程：a*x + b*y + c = x_img
        # y' 方程：d*x + e*y + f = y_img
        # [x']   [a1, b1, c1][x]
        # [y'] = [a2, b2, c2][y]
        # [1 ]   [ 0,  0,  1][1]
        # b = theta * A

        # [x'1]   [x1, y1,  1,  0,  0,  0] [a1]
        # [y'1]   [ 0,  0,  0, x1, y1,  1] [b1]
        # [...] = [.., .., .., .., .., ..] [c1]
        # [x'n]   [xn, yn,  1,  0,  0,  0] [a2]
        # [y'n]   [ 0,  0,  0, xn, yn,  1] [b2]
        #                                  [c2]
        # b = A * theta

        A.append([x_dsm, y_dsm, 1, 0, 0, 0])
        b.append(x_img)
        A.append([0, 0, 0, x_dsm, y_dsm, 1])
        b.append(y_img)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # 求解最小二乘解
    theta, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    if rank < 6:
        print("警告：矩阵秩不足，可能点共线或不足")

    # 构造3x3仿射变换矩阵
    a, b, c, d, e, f = theta
    return np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1]
    ]), residuals


def apply_transform(x, y, matrix):
    """应用仿射变换到单个点"""
    x_new = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    y_new = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return x_new, y_new

def read_img_apply_transform(img_path, matrix):
    """对读取的影像应用仿射变换测试点"""
    # x方向平移量
    x_ = matrix[0, 2]

    # 原始影像处理
    ori_img = cv2.imread(img_path, -1)
    height, width = ori_img.shape[:2]

    y_mgrid, x_mgrid = np.mgrid[0:height, 0:width]

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_mgrid)
    coords = np.stack([x_mgrid, y_mgrid, ones], axis=0)

    # 计算逆变换矩阵
    inv_matrix = np.linalg.inv(matrix)

    # 应用逆变换，找到原始DSM中的对应坐标
    src_coords = np.tensordot(inv_matrix, coords, axes=1)  # 形状 (3, height, width)
    src_x, src_y = src_coords[0], src_coords[1]  # 提取x和y坐标

    # 使用双线性插值从原始DSM中获取值
    transformed_dsm = map_coordinates(ori_img, [src_y, src_x], order=0, mode='constant', cval=-999)
    transformed_dsm[transformed_dsm == 0] = -999

    # transformed_dsm_ = np.ones_like(transformed_dsm) * -999
    # if x_ < 0:
    #     transformed_dsm_[:, :int(x_)] = transformed_dsm[:, -int(x_):]

    return transformed_dsm



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




# 示例使用
if __name__ == "__main__":
    rect_points_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/epi_res/all-GF01_PA1_041692_20210119_MY150_01_012_L1A_01-GF06_PAN_013133_20201108_MY351_01_036_L1A_01/points.txt"
    ori_disp_path = r'/home/dshare/01Data/3DDisp/LiDAR/NJ/epi_res/all-GF01_PA1_041692_20210119_MY150_01_012_L1A_01-GF06_PAN_013133_20201108_MY351_01_036_L1A_01/epiDisp.tif'
    save_disp_path = r'/home/dshare/01Data/3DDisp/LiDAR/NJ/epi_res/all-GF01_PA1_041692_20210119_MY150_01_012_L1A_01-GF06_PAN_013133_20201108_MY351_01_036_L1A_01/rect_epiDisp.tif'

    # 1. 读取点文件
    print("\n")
    print("*" * 30)
    print("=1= 读取匹配点文件")
    try:
        points = read_points(rect_points_path)
    except FileNotFoundError:
        print("错误：文件不存在")
        exit()

    # 2. 计算变换矩阵
    print("\n")
    print("*" * 30)
    print("=2= 计算变换矩阵")
    try:
        affine_matrix, residuals = compute_affine_matrix(points)
    except ValueError as e:
        print(e)
        exit()

    print("仿射变换矩阵：")
    print(affine_matrix)
    print("仿射变换残差：" + str(residuals))

    # 3. 验证第一个点
    print("\n")
    print("=3= 验证其中一个点")
    if len(points) > 0:
        x, y, x_t, y_t = points[0]
        pred_x, pred_y = apply_transform(x, y, affine_matrix)
        print(f"\n验证第一个点：")
        print(f"原始坐标：({x}, {y})")
        print(f"预测坐标：({pred_x:.2f}, {pred_y:.2f})")
        print(f"实际坐标：({x_t}, {y_t})")
        print(f"残差：X误差 {abs(pred_x - x_t):.2f}, Y误差 {abs(pred_y - y_t):.2f}")

    # 4. 对原始图像进行处理
    print("\n")
    print("*" * 30)
    print("=4= 读取原始图像进行变换")
    transform_img = read_img_apply_transform(ori_disp_path, affine_matrix)

    # 5. 保存校正后的结果
    print("\n")
    print("*" * 30)
    print("=5= 保存校正后的结果")
    cv2.imwrite(save_disp_path, transform_img)

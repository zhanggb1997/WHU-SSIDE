'''
Project    : RSDeploy
FileName   : points_rectify .py
CreateTime : 2025/3/17 
=======================
@CopyRight : WHU-
@Author    : zhang
@Contact   : zhanggb1997@163.com
@Content   : #  #
'''
import cv2
import laspy
import numpy as np


from disp_rectify import read_points


def calculate_residuals(affine_matrix, points):
    xs_las, ys_las, xs_img, ys_img = np.array(points).T

    xyo_gt = np.concatenate([xs_img[:,None], ys_img[:,None]], axis=1)

    xyo = np.concatenate([xs_las[:,None], ys_las[:,None], np.ones_like(xs_las)[:,None]], axis=1)

    xyo_trans = np.dot(affine_matrix, xyo.T)[:2]

    residuals = xyo_trans.T - xyo_gt
    rmse = np.sqrt(np.mean(residuals**2, axis=0))


    return residuals, rmse



def generate_polynomial_features(XYZ, degree=3):
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    features = []

    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            features.append((X ** i) * (Y ** j))

    features.append(Z)

    return np.column_stack(features)


def compute_affine_matrix(points):
    if len(points) < 3:
        raise ValueError("至少需要3个非共线点以计算仿射变换")

    # 构建最小二乘方程组 A * theta = b
    A = []
    b = []
    for x_pot, y_pot, x_img, y_img in points:
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

        A.append([x_pot, y_pot, 1, 0, 0, 0])
        b.append(x_img)
        A.append([0, 0, 0, x_pot, y_pot, 1])
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
    x_new = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    y_new = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return x_new, y_new


def xy_apply_transform(x_, y_, matrix):

    ones = np.ones_like(x_).astype(np.float32)
    coords = np.stack([x_, y_, ones], axis=0).astype(np.float32)
    new_coords = np.tensordot(matrix, coords, axes=1).astype(np.float32)
    new_x, new_y = new_coords[0], new_coords[1]

    return new_x, new_y

def points_apply_transform(las, matrix):
    x_ = las.x
    y_ = las.y

    ones = np.ones_like(x_).astype(np.float32)
    coords = np.stack([x_, y_, ones], axis=0)

    new_coords = np.tensordot(matrix, coords, axes=1).astype(np.float32)
    new_x, new_y = new_coords[0], new_coords[1]

    from points_process import SelfLas
    transformed_las = SelfLas()
    transformed_las.x = new_x.astype(np.float32)
    transformed_las.y = new_y.astype(np.float32)
    transformed_las.z = las.z.astype(np.float32)
    return transformed_las






if __name__ == '__main__':
    pass

'''
Project    : RSDeploy
FileName   : points_rectify .py
CreateTime : 2025/3/17 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import cv2
import laspy
import numpy as np


from disp_rectify import read_points


def calculate_residuals(affine_matrix, points):
    """残差计算"""
    xs_las, ys_las, xs_img, ys_img = np.array(points).T

    xyo_gt = np.concatenate([xs_img[:,None], ys_img[:,None]], axis=1)

    xyo = np.concatenate([xs_las[:,None], ys_las[:,None], np.ones_like(xs_las)[:,None]], axis=1)

    xyo_trans = np.dot(affine_matrix, xyo.T)[:2]

    residuals = xyo_trans.T - xyo_gt
    rmse = np.sqrt(np.mean(residuals**2, axis=0))

    print("X方向最大残差:", np.max(np.abs(residuals[:, 0])))
    print("Y方向最大残差:", np.max(np.abs(residuals[:, 1])))
    print("X方向平均残差:", np.mean(np.abs(residuals[:, 0])))
    print("Y方向平均残差:", np.mean(np.abs(residuals[:, 1])))
    print("X方向RMSE:", rmse[0])
    print("Y方向RMSE:", rmse[1])

    return residuals, rmse



def points_apply_3d_polynomial_transform(las, matrix, degree):
    """对读取的影像应用仿射变换测试点"""

    # 原始点云数据坐标位置
    x_ = las.x
    y_ = las.y
    z_ = las.z
    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_)

    A = generate_polynomial_features(np.array([x_, y_, z_]).T, degree)

    # # 计算逆变换矩阵
    # inv_matrix = np.linalg.inv(matrix)

    # 找到原始DSM中的对应坐标
    src_coords = np.tensordot(matrix, A.T, axes=1)  # 形状 (3, height, width)
    src_x, src_y = src_coords[0], src_coords[1]  # 提取x和y坐标

    las.x = src_x
    las.y = src_y

    transformed_las = las

    return transformed_las



def generate_polynomial_features(XYZ, degree=3):
    """生成带高程补偿的多项式"""
    """
    生成XY多项式特征（最高到degree阶）和Z一阶项
    :param X: Nx1数组，LiDAR的X坐标
    :param Y: Nx1数组，LiDAR的Y坐标
    :param Z: Nx1数组，LiDAR的Z坐标
    :param degree: XY多项式阶数（1, 2, 3）
    :return: 特征矩阵（N x (n_features + 1)）
    """
    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    features = []

    # 添加XY多项式项
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            features.append((X ** i) * (Y ** j))

    # 添加Z的一阶项
    features.append(Z)

    return np.column_stack(features)

def ordinary_least_squares(A, b, lambda_=1e-6):
    """
    普通最小二乘法求解 Ax = b
    :param A: 设计矩阵（N x M）
    :param b: 目标变量（N x 1）
    :return: 系数向量x（M x 1）
    """
    # 计算 (A^T A)^{-1} A^T b
    ATA = np.dot(A.T, A)  # (M x N) @ (N x M) → M x M

    # 生成正则化矩阵（λ乘以单位矩阵）
    regularization = lambda_ * np.eye(ATA.shape[0])  # M x M

    ATb = np.dot(A.T, b)

    x = np.linalg.solve(ATA + regularization, ATb)  # 使用求解器提高稳定性
    return x


def compute_3d_polynomial_matrix(points, degree=3):
    XYZ = points[:, :3]
    img_x = points[:, 3]
    img_y = points[:, 4]

    A = generate_polynomial_features(XYZ, degree=degree)

    # 求解x'方向的系数
    coefficients_x = ordinary_least_squares(A, img_x)
    # 求解y'方向的系数
    coefficients_y = ordinary_least_squares(A, img_y)
    # 求解z'方向的系数
    coefficients_z = np.zeros_like(coefficients_x)
    coefficients_z[-1] = 1

    # 构造变换矩阵
    return np.array([
        coefficients_x,
        coefficients_y,
        coefficients_z
    ])


# class ElevationAwareFeatureGenerator:
#     """生成带高程补偿的多项式"""
#
#     def __init__(self, degree=2):
#         self.degree = degree
#
#     def fit_transform(self, X):
#         """
#         X: 归一化后的输入 [X,Y,Z]
#         输出特征矩阵包含：
#         - XY多项式项（最高到degree阶）
#         - Z一阶项
#         """
#         X = X.copy()
#         n_samples = X.shape[0]
#         features = []
#
#         # XY多项式项
#         x = X[:, 0]
#         y = X[:, 1]
#         for d in range(1, self.degree + 1):
#             for i in range(d + 1):
#                 j = d - i
#                 features.append((x ** i) * (y ** j))
#
#         # Z一阶项
#         z = X[:, 2]
#         features.append(z)
#
#         return np.array(features).T
#
#     @property
#     def n_features(self):
#         return sum(d + 1 for d in range(1, self.degree + 1)) + 1
#
#
# class ElevationAwareRegistration:
#     """高程感知配准模型"""
#
#     def __init__(self, degree=2, alpha=0.1):
#         self.degree = degree
#         self.x_model = make_pipeline(
#             StandardScaler(),
#             Ridge(alpha=alpha, fit_intercept=True)
#         )
#         self.y_model = make_pipeline(
#             StandardScaler(),
#             Ridge(alpha=alpha, fit_intercept=True)
#         )
#         self.feature_gen = ElevationAwareFeatureGenerator(degree)
#
#     def fit(self, lidar_xyz, image_xy):
#         # 生成特征
#         features = self.feature_gen.fit_transform(lidar_xyz)
#
#         # 分别训练x/y模型
#         self.x_model.fit(features, image_xy[:, 0])
#         self.y_model.fit(features, image_xy[:, 1])
#
#     def predict(self, lidar_xyz):
#         features = self.feature_gen.fit_transform(lidar_xyz)
#         return np.column_stack((
#             self.x_model.predict(features),
#             self.y_model.predict(features)
#         ))


def compute_affine_matrix(points):
    """
    使用最小二乘法计算仿射变换矩阵
    """
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



def compute_3d_affine_matrix(points):
    """
    使用最小二乘法计算三维仿射变换矩阵
    control_points: [[lidar_x, lidar_y, lidar_z, image_x, image_y], ...]
    """
    if len(points) < 3:
        raise ValueError("至少需要3个非共线点以计算仿射变换")

    # 构建最小二乘方程组 A * theta = b
    A = []
    b = []
    for x_pot, y_pot, z_pot, x_img, y_img in points:
        # [x'1]   [x1, y1,  1,  0,  0,  0] [a1]
        # [y'1]   [ 0,  0,  0, x1, y1,  1] [b1]
        # [...] = [.., .., .., .., .., ..] [c1]
        # [x'n]   [xn, yn,  1,  0,  0,  0] [a2]
        # [y'n]   [ 0,  0,  0, xn, yn,  1] [b2]
        #                                  [c2]
        # b = A * theta

        A.append([x_pot, y_pot, z_pot, 1, 0, 0, 0, 0])
        b.append(x_img)
        A.append([0, 0, 0, 0, x_pot, y_pot, z_pot, 1])
        b.append(y_img)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # 求解最小二乘解
    theta, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    if rank < 6:
        print("警告：矩阵秩不足，可能点共线或不足")

    # 构造3x3仿射变换矩阵
    a, b, c, d, e, f, g, h = theta
    return np.array([
        [a, b, c, d],
        [e, f, g, h],
        [0, 0, 0, 1]
    ]), residuals


def apply_transform(x, y, matrix):
    """应用仿射变换到单个点"""
    x_new = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    y_new = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return x_new, y_new


def xy_apply_transform(x_, y_, matrix):
    """对读取的影像应用仿射变换测试点"""

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_).astype(np.float32)
    coords = np.stack([x_, y_, ones], axis=0).astype(np.float32)

    # # 计算逆变换矩阵
    # inv_matrix = np.linalg.inv(matrix)

    # 应用逆变换，找到原始DSM中的对应坐标
    new_coords = np.tensordot(matrix, coords, axes=1).astype(np.float32)  # 形状 (3, height, width)
    new_x, new_y = new_coords[0], new_coords[1]  # 提取x和y坐标

    return new_x, new_y

def points_apply_transform(las, matrix):
    """对读取的影像应用仿射变换测试点"""

    # 原始点云数据坐标位置
    x_ = las.x
    y_ = las.y

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_).astype(np.float32)
    coords = np.stack([x_, y_, ones], axis=0)

    # # 计算逆变换矩阵
    # inv_matrix = np.linalg.inv(matrix)

    # 应用逆变换，找到原始DSM中的对应坐标
    new_coords = np.tensordot(matrix, coords, axes=1).astype(np.float32)  # 形状 (3, height, width)
    new_x, new_y = new_coords[0], new_coords[1]  # 提取x和y坐标

    from points_process import SelfLas
    transformed_las = SelfLas()
    transformed_las.x = new_x.astype(np.float32)
    transformed_las.y = new_y.astype(np.float32)
    transformed_las.z = las.z.astype(np.float32)
    return transformed_las

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


def points_apply_3d_transform(las, matrix):
    """对读取的影像应用仿射变换测试点"""

    # 原始点云数据坐标位置
    x_ = las.x
    y_ = las.y
    z_ = las.z

    # 将目标影像的坐标转换为齐次坐标
    ones = np.ones_like(x_)
    coords = np.stack([x_, y_, z_, ones], axis=0)

    # # 计算逆变换矩阵
    # inv_matrix = np.linalg.inv(matrix)

    # 应用逆变换，找到原始DSM中的对应坐标
    new_coords = np.tensordot(matrix, coords, axes=1)  # 形状 (3, height, width)
    new_x, new_y = new_coords[0], new_coords[1]  # 提取x和y坐标

    # 创建新文件头
    new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)

    # 写入新文件
    new_las = laspy.LasData(new_header)
    new_las.header.offsets = [0, 0, 0]
    new_las.points.offsets = [0, 0, 0]
    new_las.header.scales = [0.01, 0.01, 0.001]  # 像素坐标系
    new_las.points.scales = [0.01, 0.01, 0.001]  # 像素坐标系使用更低精度
    for dim in las.point_format.dimensions:
        new_las[dim.name] = las[dim.name]
    new_las.xyz = np.array([new_x, new_y, las.z]).astype(np.float64).transpose(1, 0)
    new_las.x = np.array(new_x).astype(np.float64)
    new_las.y = np.array(new_y).astype(np.float64)
    new_las.z = las.z

    return new_las


def las_apply_rectify(las, points_path):
    # 1. 读取点文件
    try:
        points = read_points(points_path)
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
    print("=4= 读取原始las进行变换")
    transform_las = points_apply_transform(las, affine_matrix)

    return transform_las



if __name__ == '__main__':
    pass

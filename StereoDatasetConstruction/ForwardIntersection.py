'''
Project    : RSDeploy
FileName   : ForwardIntersection .py
CreateTime : 2025/3/5 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
from scipy.optimize import least_squares
from StereoDatasetConstruction.image.img_info import RSImage


def back_projection_error(Img1Path, Img2Path, RPC1Path, RPC2Path):
    # 影像读取
    Img1 = RSImage(Img1Path, RPC1Path)
    Img2 = RSImage(Img2Path, RPC2Path)

    # 左右影像同名像点
    # pointLx, pointLy = 14464.0176, 3255.3304
    # pointRx, pointRy = 32525.4716, 8637.7806
    pointLx, pointLy = 17698.1131, 1694.9995
    pointRx, pointRy = 33894.4907, 7956.1022

    # 初始化猜测（左影像反解）
    h0 = 0.0
    lon0, lat0 = Img1.rpc_model.localization(pointLx, pointLy, h0)

    # 定义误差函数
    def error_func(params):
        lat, lon, h = params
        x1, y1 = (pointLx, pointLy)
        x2, y2 = (pointRx, pointRy)
        x1_pred, y1_pred = Img1.rpc_model.projection(lon, lat, h)
        x2_pred, y2_pred = Img2.rpc_model.projection(lon, lat, h)
        return [x1_pred - x1, y1_pred - y1, x2_pred - x2, y2_pred - y2]

    # 优化参数
    result = least_squares(
        error_func,
        [lat0, lon0, h0],
        method="lm",
        ftol=1e-6,
        max_nfev=100
    )

    # 输出结果
    lat_opt, lon_opt, h_opt = result.x
    print(f"物方点坐标：纬度={lat_opt:.6f}°, 经度={lon_opt:.6f}°, 高程={h_opt:.2f}米")

    x1_pred_, y1_pred_ = Img1.rpc_model.projection(lon_opt, lat_opt, h_opt)
    x2_pred_, y2_pred_ = Img2.rpc_model.projection(lon_opt, lat_opt, h_opt)

    x1_error = (x1_pred_ - pointLx)
    y1_error = (y1_pred_ - pointLy)
    x2_error = (x2_pred_ - pointRx)
    y2_error = (y2_pred_ - pointRy)

    print(x1_error)
    print(y1_error)
    print(x2_error)
    print(y2_error)


if __name__ == '__main__':
    imgL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_PM1_041907_20220521_KS450_01_012_L1A_01_202502180000160001/GF02_PA1_041907_20220521_KS450_01_012_L1A_01.tif"
    rpcL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_PM1_041907_20220521_KS450_01_012_L1A_01_202502180000160001/GF02_PA1_041907_20220521_KS450_01_012_L1A_01_ba_rpc1.txt"

    imgR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_202502170000260001/GF06_PAN_014166_20210117_MY261_01_037_L1A_01.tif"
    rpcR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_202502170000260001/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_ba_rpc1.txt"

    back_projection_error(imgL_path, imgR_path, rpcL_path, rpcR_path)
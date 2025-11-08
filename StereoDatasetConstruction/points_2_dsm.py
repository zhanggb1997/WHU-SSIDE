'''
Project    : RSDeploy
FileName   : points_2_dsm .py
CreateTime : 2025/5/7 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import csv
import gc
import math
import os.path
import pylas
import cv2
import laspy
import numpy as np
import pyproj
import rasterio
from pyproj import CRS, Transformer
from rasterio import float32
from rasterio._warp import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from points_process import points_Z_IQR_filter, las_voxel_max_sampling, SelfLas, laspy_voxel_max_sampling


def get_pixel_value(image_path, lon, lat):
    """
    根据地理坐标 (经度, 纬度) 提取遥感影像的像素值
    """
    with rasterio.open(image_path) as src:
        # 检查坐标系是否匹配（若需要，进行坐标转换）
        if src.crs.to_epsg() != 4326:  # 假设输入坐标是 WGS84 (EPSG:4326)
            # 使用 pyproj 转换坐标到影像的坐标系
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat

        # 将地理坐标转换为像素行列号
        row, col = src.index(x, y)

        # 读取像素值（支持多波段）
        value = src.read(1, window=((row, row + 1), (col, col + 1)))  # 读取第一个波段
        return value[0][0]


def point_init(las_paths):
    if isinstance(las_paths, list):
        pass
    else:
        las_paths = [las_paths]

    all_las = []

    # ============= 分块读取las =============
    for las_path in las_paths:
        with laspy.open(las_path) as reader:
            # 分块读取
            for points in reader.chunk_iterator(1000000):
                # mask = laspy_voxel_max_sampling(points, 0.5)
                # las_f = points[mask]
                # all_las.append(las_f)
                all_las.append(points)

    #
    # # ============= 合并las并剔除重复区域 =============
    # # temp_las = laspy.read(las_path)
    # merged_las = laspy.LasData(reader.header)
    #
    # # 合并点数据
    # merged_points_array = np.concatenate([las.array for las in all_las])
    # merged_las.points.array = merged_points_array
    # merged_points_x = np.concatenate([las.x for las in all_las])
    # merged_points_y = np.concatenate([las.y for las in all_las])
    # merged_points_z = np.concatenate([las.z for las in all_las])
    #
    # merged_las.x = merged_points_x
    # merged_las.y = merged_points_y
    # merged_las.z = merged_points_z
    #
    # # 复制额外属性
    # for dim in reader.header.point_format.dimensions:
    #     if dim.name not in ["X", "Y", "Z", 'classification']:
    #         merged_data = np.concatenate([las[dim.name] for las in all_las])
    #         merged_las[dim.name] = merged_data
    #
    # # # 剔除重复数据
    # # mask = laspy_voxel_max_sampling(merged_las, 1)
    # # merged_las_f = merged_las[mask]
    #
    # # # ============= 写入las数据 =============
    # # with laspy.open(r"/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_LosAngeles5_10_max/all_las.las", mode='w', header=merged_las.header) as writer:
    # #     writer.write_points(merged_las.points)


    merged_points_x = np.concatenate([las.x for las in all_las]).astype(np.float32)
    merged_points_y = np.concatenate([las.y for las in all_las]).astype(np.float32)
    merged_points_z = np.concatenate([las.z for las in all_las]).astype(np.float16)
    merged_las = SelfLas()
    merged_las.x = merged_points_x
    merged_las.y = merged_points_y
    merged_las.z = merged_points_z


    del all_las
    gc.collect()

    return merged_las




    # # 读取所有数据，初步筛除高程不符合区域
    # for las_path in las_paths:
    #     #  ============= 直接读取las=============
    #     # 读取所有las
    #     try:
    #         # las = pylas.read(las_path)
    #         las = laspy.read(las_path)
    #     except:
    #         print("未能有效读取{}".format(las_path))
    #     # all_las.append(las)
    #
    #     #  ============= 筛除las中异常高程 ===============
    #     # Z-Score
    #     # _, mask_z = points_Z_zscore_filter(las, z_threshold=3.0)
    #     # # IQR
    #     # mask_z = points_Z_IQR_filter(las, iqr_scale=0.02)
    #     # las_f = SelfLas()
    #     # las_f.x = np.array(las.x)[mask_z]
    #     # las_f.y = np.array(las.y)[mask_z]
    #     # las_f.z = np.array(las.z)[mask_z]
    #     las_f = SelfLas()
    #     las_f.x = np.array(las.x)
    #     las_f.y = np.array(las.y)
    #     las_f.z = np.array(las.z)
    #
    #     del las
    #     gc.collect()
    #
    #     # # #  ============= las体素格降采样 ===============
    #     las_d = las_voxel_max_sampling(las_f, 1)
    #
    #     all_las.append(las_d)


    # # 基于头信息，创建合并后的LAS对象
    # merged_las = laspy.LasData(all_las[0].header)

    # # 合并点数据
    # merged_points_array = np.concatenate([las.points.array for las in all_las])
    # merged_las.points.array = merged_points_array
    # merged_points_x = np.concatenate([las.xyz[:, 0] for las in all_las])
    # merged_points_y = np.concatenate([las.xyz[:, 1] for las in all_las])
    # merged_points_z = np.concatenate([las.xyz[:, 2] for las in all_las])
    # merged_las.x = merged_points_x
    # merged_las.y = merged_points_y
    # merged_las.z = merged_points_z

    # # 更新信息
    # merged_header = merged_las.header
    # merged_header.mins = [
    #     min([las.header.x_min for las in all_las]),
    #     min([las.header.y_min for las in all_las]),
    #     min([las.header.z_min for las in all_las]),
    # ]
    # merged_header.maxs = [
    #     max([las.header.x_max for las in all_las]),
    #     max([las.header.y_max for las in all_las]),
    #     max([las.header.z_max for las in all_las]),
    # ]
    # merged_header.x_min = min([las.header.x_min for las in all_las])
    # merged_header.x_max = max([las.header.x_max for las in all_las])
    # merged_header.y_min = min([las.header.y_min for las in all_las])
    # merged_header.y_max = max([las.header.y_max for las in all_las])
    # merged_header.z_min = min([las.header.z_min for las in all_las])
    # merged_header.z_max = max([las.header.z_max for las in all_las])
    #
    # # 复制额外属性
    # for dim in all_las[0].point_format.dimensions:
    #     if dim.name not in ["X", "Y", "Z"]:
    #         merged_data = np.concatenate([las[dim.name] for las in all_las])
    #         merged_las[dim.name] = merged_data
    #
    # las_res = merged_las

    # merged_points_x = np.concatenate([las.x for las in all_las]).astype(np.float64)
    # merged_points_y = np.concatenate([las.y for las in all_las]).astype(np.float64)
    # merged_points_z = np.concatenate([las.z for las in all_las]).astype(np.float32)
    # merged_las = SelfLas()
    # merged_las.x = merged_points_x
    # merged_las.y = merged_points_y
    # merged_las.z = merged_points_z
    #
    #
    # del all_las
    # gc.collect()
    #
    # return merged_las


def point_localization(las, save_path=None):
    # # las头信息
    # header = las.header
    #
    # # 尝试解析坐标参考系统（CRS）
    # crs = None
    #
    # # 检查头中的CRS信息（LASpy 2.4+）
    # if hasattr(header, "parse_crs"):
    #     crs = header.parse_crs()
    #     # if crs:
    #     #     print("\n坐标参考系统（CRS）信息:")
    #     #     print(crs)
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

    # # 输出CRS详情
    # if crs:
    #     print(f"CRS名称: {crs.name}")
    #     print(f"坐标类型: {'地理坐标系' if crs.is_geographic else '投影坐标系'}")
    #     print(f"EPSG代码: {crs.to_epsg() if crs.to_epsg() else '未知'}")

    # # 转换坐标到地理坐标系（示例）
    # if crs.to_epsg() and not crs.is_geographic:
    #     # 创建转换器（假设目标为WGS84，EPSG:4326）
    #     target_crs = CRS.from_epsg(4326)
    #     transformer = Transformer.from_crs(crs, target_crs)
    #
    # elif crs and not crs.is_geographic:  # America
    # 创建转换器（假设目标为WGS84，EPSG:4326）
    # source_crs = CRS.from_epsg(6455).to_3d()  # 转换为3D坐标系
    # source_crs = CRS.from_epsg(6318).to_3d()  # 转换为3D坐标系
    # source_crs = crs  # 转换为3D坐标系
    # target_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_epsg(32616)  # UTM坐标系 cook
    # target_crs = CRS.from_epsg(32610)  # UTM坐标系 losangles
    # target_crs = CRS.from_epsg(4979)  # EPSG:4979 = WGS84 (3D, 经纬度 + 椭球高)
    # transformer = Transformer.from_crs("EPSG:6455+5703", "EPSG:4326+3855")  # 输入：6455 + NAVD88 北美垂直基准   输出：WGS84 + 椭球高
    # transformer = Transformer.from_crs(source_crs,target_crs, always_xy=True)   # 启用大地水准面校正
    transformer = Transformer.from_crs("EPSG:6455", target_crs, always_xy=True)  # 输入：6455 + NAVD88 北美垂直基准   输出：WGS84 + 椭球高

    # else:
    #     # 南京地区NJ预先定义的CRS
    #     target_crs = pyproj.CRS("EPSG:4326")  # EPGS:4326 代码代表 WGS-84坐标系
    #     # 定义UTM坐标系（基于WGS84）确定对应的UTM区域
    #     NJ_center_zone = int(118 / 6) + 31  # UTM区域
    #     utm_proj = pyproj.CRS(f"EPSG:326{NJ_center_zone}")  # EPSG:326 代码代表 UTM坐标系
    #     transformer = Transformer.from_crs(utm_proj, wgs84_proj)
    #     # 高斯克吕格
    #     format = '+proj=tmerc +lat_0=0 +lon_0=' + str(-88) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    #     crs_GK = CRS.from_proj4(format)
    #     # # 转换
    #     transformer = Transformer.from_crs(crs_GK, target_crs)

    # # # 访问点坐标
    # print("\n前三点的坐标（X, Y, Z）:")
    # for i in range(3):
    #     x, y, z = las.x[i], las.y[i], las.z[i]
    #     lon, lat = transformer.transform(x, y)
    #     print(f"\n第{i + 1}个点的经纬度: 经度={lon:.6f}, 纬度={lat:.6f}")
    #     print(f"点{i + 1}: {las.x[i]}, {las.y[i]}, {las.z[i]}")

    # 坐标转换
    # lons, lats, heis = transformer.transform(las.x, las.y, las.z)
    lons, lats = transformer.transform(las.x, las.y)

    # 获取正高和椭球高的差异
    path_emg08 = "/home/dshare/06Test/us_nga_egm2008_1.tif"
    eval_dsip = get_pixel_value(path_emg08, np.array(lons)[0], np.array(lats)[0])

    # 高程英尺转为米，并添加偏移量处理
    heis = 0.3048 * np.array(las.z) + eval_dsip

    # # 创建新文件头
    # new_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    #
    # # 写入新文件
    # new_las = laspy.LasData(new_header)
    # new_las.header.offsets = [0, 0, 0]
    # new_las.points.offsets = [0, 0, 0]
    # new_las.header.scales = [0.01, 0.01, 0.001]  # 经纬度使用更高精度
    # new_las.points.scales = [0.01, 0.01, 0.001]  # 经纬度使用更高精度
    # for dim in las.point_format.dimensions:
    #     new_las[dim.name] = las[dim.name]
    # # new_las.xyz = np.array([lons, lats, las.z]).astype(np.float64).transpose(1,0)
    # new_las.xyz = np.array([lons, lats, heis]).astype(np.float32).transpose(1,0)
    new_las = SelfLas()
    new_las.x = lons.astype(np.float64)
    new_las.y = lats.astype(np.float64)
    new_las.z = heis.astype(np.float32)

    # # 添加CRS信息（LAS 1.4+）
    # if hasattr(new_las.header, "add_crs"):
    #     new_las.header.add_crs(target_crs)

    # 如果有保存地址的话
    if save_path:
        new_las.write(save_path)
        print(f"转换完成，文件已保存为：{save_path}")

    # 返回信息
    return new_las


def point_2_dsm(las, save_path=None):
    x = las.x
    y = las.y
    z = las.z
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 获取地理信息
    # CRS = las.header.parse_crs().geodetic_crs
    # # 创建栅格网格
    # cols = int((x_max - x_min) // resolution) + 1
    # rows = int((y_max - y_min) // resolution) + 1
    # cols = 1024
    # rows = 1024
    # cols = 512
    # rows = 512
    # # 分辨率信息
    # resolution_x = (x_max - x_min) / cols
    # resolution_y = (y_max - y_min) / rows

    # 分辨率信息
    resolution_x = 1
    resolution_y = 1
    cols = math.ceil((x_max - x_min) / resolution_x)
    rows = math.ceil((y_max - y_min) / resolution_y)

    # 使用分箱统计计算每个像素的最大高程
    dsm, col_edges, row_edges, _ = binned_statistic_2d(
        y, x, values=z,
        # statistic='min', bins=(rows, cols),
        statistic='max', bins=(rows, cols),
        # statistic='median', bins=(rows, cols),
        # statistic='mean', bins=(rows, cols),
        # range=[[0, cols], [0, rows]]
    )

    # 处理缺失值（无点的像素设为NaN）
    # dsm = np.where(np.isnan(dsm), 0, dsm)
    dsm = np.where(np.isnan(dsm), np.nan, dsm)
    dsm = np.flip(dsm, 0)

    # cv2.imwrite(save_path, dsm)

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
    transform = from_origin(x_min, y_max, resolution_x, resolution_y)
    # transform = from_origin(10., 20., resolution_x, resolution)

    # 写入GeoTIFF
    try:
        with rasterio.open(
                save_path,
                "w+",
                driver="GTiff",
                height=rows,
                width=cols,
                count=1,
                # dtype=dsm.dtype,
                dtype=float32,
                # crs=CRS,  # 指定坐标系,像素坐标系
                crs=CRS.from_epsg(32616),  # 指定坐标系 cook
                # crs=CRS.from_epsg(32610),  # 指定坐标系 losangles
                # crs=CRS.from_epsg(4326),  # 指定坐标系
                transform=transform,
                nodata=np.nan
        ) as dst:
            dst.write(dsm, 1)
    except FileExistsError:
        print("dsm.tif已经存在！")

    #
    # # 投影变换处理
    # with rasterio.open(save_path) as src:
    #     # 获取源坐标系
    #     src_crs = src.crs
    #     # src_crs = CRS.from_epsg(6455)
    #     # 定义目标坐标系
    #     dst_crs = "EPSG:4326"
    #     # 计算在新空间参考系下的仿射变换参数，图像尺寸
    #     dst_transform, dst_width, dst_height = calculate_default_transform(
    #         src_crs,  # 输入坐标系
    #         dst_crs,  # 输出坐标系
    #         src.width,  # 输入图像宽
    #         src.height,  # 输入图像高
    #         *src.bounds)  # 输入数据源的图像范围
    #
    #     # 更新数据集的元数据信息
    #     # profile = src.profile
    #     profile = src.meta.copy()
    #     profile.update({
    #         'crs': dst_crs,
    #         'transform': dst_transform,
    #         'width': dst_width,
    #         'height': dst_height
    #     })
    #
    #     # 创建输出文件并写入数据
    #     with rasterio.open(save_path, "w", **profile) as dst:
    #         src_array = src.read(1)
    #         dst_array = np.empty((dst_height, dst_width), dtype=profile['dtype'])  # 初始化输出图像数据
    #
    #         # 重投影
    #         reproject(
    #             # 源文件参数
    #             source=src_array,  # 源波段
    #             src_transform=src.transform,
    #             src_crs=src_crs,
    #             # 目标文件参数
    #             destination=dst_array,  # 目标波段
    #             dst_transform=transform,
    #             dst_crs=dst_crs,
    #             # 其它配置
    #             resampling=Resampling.nearest,  # 重采样方法（根据需求调整）
    #             num_threads=2
    #         )
    # #         dst.write(dst_array, 1)
    # dsm_ = dsm
    #
    # return dsm_






if __name__ == '__main__':
    # # # 打开txt文件
    # with open('/home/dshare/06Test/LiDAR/America/dupage-las2022/2022/tile_index/GF7_5_all.txt', 'r') as file:
    # # # with open('/home/dshare/06Test/LiDAR/America/dupage-las2022/2022/tile_index/GF7_DSM_1-1.txt', 'r') as file:
    #     # 使用csv.reader读取文件内容
    #     reader = csv.reader(file)
    #     # 跳过标题行
    #     next(reader)
    #     # 提取NAME字段的值
    #     names = [row[1] for row in reader]
    # # point_paths = ["/home/dshare/06Test/LiDAR/America/dupage-las2022/2022/las/{}.las".format(name) for name in names]
    # point_paths = ["/home/dshare/06Test/LiDAR/America/cook-las5/{}.las".format(name) for name in names]
    #
    #
    # # names = os.listdir("/home/dshare/06Test/LiDAR/America/cook-las/cook-las1")
    #
    # point_paths_, names_ = [], []
    # for name in names:
    #     # point_path = "/home/dshare/06Test/LiDAR/America/dupage-las2022/2022/las/{}.las".format(name)
    #     point_path = "/home/dshare/06Test/LiDAR/America/cook-las5/{}".format(name)
    #     if os.path.exists(point_path):
    #         point_paths_.append(point_path)
    #         names_.append(name)
    #
    # tqdm_list = tqdm(zip(point_paths_, names_), desc="processing", total=len(names_), ncols=100)
    # for point_path, point_name in tqdm_list:
    #     # point_save_name = "/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_dug10_max/1_{}.tiff".format(point_name)
    #     point_save_name = "/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_cook5_10_max/{}.tiff".format(point_name)
    #
    #     # if not os.path.exists(point_save_name):
    #     #     try:
    #     #         las = point_init(point_path)
    #     #         las_loc = point_localization(las)
    #     #         point_2_dsm(las_loc, point_save_name)
    #     #     except:
    #     #         print("{}处理失败！".format(point_path))
    #     #         continue
    #     # else:
    #     #     continue
    #     las = point_init(point_path)
    #     las_loc = point_localization(las)
    #     point_2_dsm(las_loc, point_save_name)




    # # 打开txt文件
    # names = ["all-losangle0-10", "all-losangle11-19"]
    # names = ["all-losangle11-19"]
    names = [str(i) for i in range(0, 14)]

    point_paths = ["/home/dshare/06Test/LiDAR/America/cook-las5/filter/{}.las".format(name) for name in names]

    point_save_names = ["/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_cook5_10_max/cook_dsm_{}.tiff".format(name) for name in names]

    tqdm_list = tqdm(zip(point_paths, point_save_names), desc="processing", total=len(names), ncols=100)

    for point_path, point_save_name in tqdm_list:
        las = point_init(point_path)
        # las_loc = point_localization(las)
        # point_2_dsm(las_loc, point_save_name)
        las_loc = point_localization(las)
        point_2_dsm(las_loc, point_save_name)
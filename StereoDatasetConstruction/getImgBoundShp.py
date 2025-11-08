'''
Project    : RSDeploy
FileName   : getImgBoundShp .py
CreateTime : 2025/5/12 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import rasterio
from pyproj import CRS
from rasterio.features import shapes
from shapely import unary_union
from shapely.geometry import shape, Polygon
import geopandas as gpd
from osgeo import gdal, ogr, osr

def generate_image_boundary(input_tif, output_shp):
    # 读取遥感影像
    with rasterio.open(input_tif) as src:
        # 获取影像的地理范围（左下、右下、右上、左上）
        bounds = src.bounds
        crs = src.crs

    # 定义矩形边界坐标点（顺序：左下→右下→右上→左上→左下）
    coords = [
        (bounds.left, bounds.bottom),
        (bounds.right, bounds.bottom),
        (bounds.right, bounds.top),
        (bounds.left, bounds.top),
        (bounds.left, bounds.bottom)
    ]

    # 创建多边形几何体
    polygon = Polygon(coords)

    # # 获取影像的掩膜（非空像素区域）
    # mask = src.dataset_mask()
    # # 提取所有多边形的几何形状
    # geometries = []
    # for geom, val in shapes(mask, transform=src.transform):
    #     if val == 0:  # 跳过无效区域（假设空值为0）
    #         continue
    #     geometries.append(shape(geom))
    #
    # # 合并所有多边形（处理影像可能存在多个不连续区域）
    # boundary = unary_union(geometries)

    # 创建 GeoDataFrame
    # gdf = gpd.GeoDataFrame(geometry=[polygon], crs=src.crs)
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=CRS.from_epsg(4326))

    # 保存为 Shapefile
    gdf.to_file(output_shp, driver='ESRI Shapefile')
    print(f"边界已保存至: {output_shp}")

#
# def generate_image_boundary(input_tif, output_shp):
#     # 打开影像文件
#     ds = gdal.Open(input_tif)
#     if ds is None:
#         raise ValueError("无法打开影像文件: " + input_tif)
#
#     # 获取影像坐标系
#     src_crs = osr.SpatialReference()
#     src_crs.ImportFromWkt(ds.GetProjectionRef())
#     if src_crs.IsProjected() == 0 and src_crs.IsGeographic() == 0:
#         raise ValueError("影像未定义坐标系，需手动指定！")


if __name__ == '__main__':
    # 输入影像
    input_image_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-BWDPAN.tiff"
    # 输出shp
    output_bound_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-BWDPAN.shp"
    # 示例调用
    generate_image_boundary(input_image_path, output_bound_path)
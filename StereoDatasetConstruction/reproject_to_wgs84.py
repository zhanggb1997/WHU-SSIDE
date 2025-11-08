'''
Project    : RSDeploy
FileName   : reproject_to_wgs84 .py
CreateTime : 2025/5/24 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_to_wgs84(input_path, output_path):
    # 打开源UTM影像
    with rasterio.open(input_path) as src:
        # 计算目标坐标系参数（自动识别输入CRS）
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,          # 输入CRS（UTM Zone 16N，如EPSG:32616）[8](@ref) 32610 losangle
            'EPSG:4326',      # 目标CRS（WGS84地理坐标系）
            src.width,        # 原影像宽度
            src.height,       # 原影像高度
            *src.bounds       # 原影像地理边界
        )

        # 更新输出影像元数据
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': 'EPSG:4326',
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': src.nodata  # 继承原影像的无效值设置
        })

        # 执行重投影
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            for band in range(1, src.count + 1):  # 支持多波段影像
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.bilinear  # 双线性重采样[6](@ref)
                )

# 调用示例
reproject_to_wgs84('/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_cook5_10_max/all.tif',
                   '/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_cook5_10_max/all_wgs84.tif')
# reproject_to_wgs84('/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_all/merged10_output_max_A-all.tif',
#                    '/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_all/merged10_output_max_A-all_wgs84.tif')
# reproject_to_wgs84('/home/dshare/01Data/3DDisp/LiDAR/America/dom/google_dom.tif',
#                    '/home/dshare/01Data/3DDisp/LiDAR/America/dom/google_dom_wgs84.tif')
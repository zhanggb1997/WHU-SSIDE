'''
Project    : RSDeploy
FileName   : tiff_merge .py
CreateTime : 2025/5/21 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import numpy as np
import rasterio
from rasterio.merge import merge
import glob
from osgeo import gdal, gdalconst
from rasterio.warp import calculate_default_transform


def merge_tifs(input_files, output_path, method='first'):

    # # 设置参数：统一投影、重采样方法、无数据值
    # options = gdal.WarpOptions(
    #     srcSRS='EPSG:32616',  # 输入文件投影（需统一）
    #     dstSRS='EPSG:4326',  # 输出投影
    #     resampleAlg=gdal.GRA_NearestNeighbour,  # 重采样算法
    #     srcNodata=np.nan,  # 输入无数据值
    #     dstNodata=np.nan,  # 输出无数据值
    #     format="GTiff",
    #     # creationOptions=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"]  # 压缩与大文件支持
    #     outputType=gdalconst.GDT_Float32
    # )
    #
    # # 执行拼接
    # gdal.Warp(output_path, input_files, options=options)


    # 打开所有源文件
    srcs = [rasterio.open(file, mode="r") for file in input_files]

    try:
        # 检查所有文件的CRS和分辨率是否一致
        crs_list = [src.crs for src in srcs]
        if len(set(crs_list)) > 1:
            raise ValueError("输入文件的坐标参考系统(CRS)不一致，请先统一CRS。")

        # 检查分辨率是否一致
        resolution_set = set((src.transform.a, abs(src.transform.e)) for src in srcs)
        if len(resolution_set) > 1:
            raise ValueError("输入文件的分辨率不一致，请先统一分辨率。")

        # #
        # for src in srcs:
        #     transform, width, height = calculate_default_transform(
        #         src.crs, dst_crs, src.width, src.height, *src.bounds
        #     )
        # # 处理边界
        # left, bottom, right, top = srcs[0].bounds
        # for src in srcs[1:]:
        #     left_, bottom_, right_, top_ = src.bounds
        #     if left_ < left: left = left_
        #     if top_ < top: top = top_
        #     if bottom_ > bottom: bottom = bottom_
        #     if right_ > right: right = right_

        # 合并影像，method可选：'first', 'last', 'min', 'max', 'sum', 'average'
        # merged_data, merged_transform = merge(srcs, bounds=(left, top, right, bottom), method=method)
        merged_data, merged_transform = merge(srcs, method=method)

        # 准备输出元数据
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
            "nodata": srcs[0].nodata  # 继承第一个文件的nodata值
        })

        # 写入输出文件
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(merged_data)
        print(f"合并完成，结果已保存至：{output_path}")

    finally:
        # 确保所有文件被关闭
        for src in srcs:
            src.close()
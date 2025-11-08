'''
Project    : StereoDatasetConstruction
FileName   : main .py
CreateTime : 2025/1/11 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import gc
import os.path
import cv2
import numpy as np

from disp_rectify import img1_apply_transform
from epipolar_check import get_good_sifts
from points_process import LasInit, point_localization, point_projection, point_2_dsm, point_rectify, las_reproject1L, las_reproject1R
from points_rectify import compute_affine_matrix, points_apply_transform
from get_intersection_region import get_epipolar_coeffs_tonggui, norm_save_epipolar_image, resample_epipolar_imageL_tonggui, resample_epipolar_imageR_tonggui,  get_disp_res_tonggui3
from image.img_info import RSImage


def main_stereo_tonggui(point_paths, imgL_path, rpcL_path, xmlL_path, imgR_path, rpcR_path, xmlR_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # 步骤1 点云处理
    print("="*10 + str(1) + "="*10 + "\n点云读取/拼接")
    las_all = LasInit(point_paths)

    # 步骤2 影像/RPC/XML读取
    print("="*10 + str(2) + "="*10 + "\n影像信息读取")
    RSImg1 = RSImage(imgL_path, rpcL_path, xmlL_path)
    RSImg2 = RSImage(imgR_path, rpcR_path, xmlR_path)

    # # 步骤3 点云的投影坐标系转换为大地坐标系WGS84
    print("="*10 + str(3) + "="*10 + "\n点云坐标系转换")
    las_local = point_localization(las_all)
    del las_all
    gc.collect()

    # 步骤4 点云基于RPCModel进行投影由大地坐标系WGS84转为像素坐标系
    print("="*10 + str(4) + "="*10 + "\n点云投影至影像1")
    las_pixel1 = point_projection(las_local, RSImg1.rpc_model)
    print("="*10 + str(4) + "="*10 + "\n点云投影至影像2")
    las_pixel2 = point_projection(las_local, RSImg2.rpc_model)
    del las_local
    gc.collect()

    # # # 步骤5 点云的变换矩阵转换至影像
    print("="*10 + str(5) + "="*10 + "\n点云1仿射变换")
    rec_las_pixel1 = point_rectify(las_pixel1, RSImg1, match_split=5, save_path_dsm=save_path+"/dsm1.tif", save_path_img=save_path+"/img1.tif", read_path_points=save_path+"/points1.txt", save_path_affine_matrix=save_path+"/affine_matrix1.txt")
    del las_pixel1
    print("="*10 + str(5) + "="*10 + "\n点云2仿射变换")
    rec_las_pixel2 = point_rectify(las_pixel2, RSImg2, match_split=5, save_path_dsm=save_path+"/dsm2.tif", save_path_img=save_path+"/img2.tif", read_path_points=save_path+"/points2.txt", save_path_affine_matrix=save_path+"/affine_matrix2.txt")
    del las_pixel2
    gc.collect()


    # 步骤6 交会区域像素坐标边界确定
    print("="*10 + str(6) + "="*10 + "\n交会区域像素坐标边界确定")
    yy_range1_ = (max(0, np.min(rec_las_pixel1.y)), min(RSImg1.height, np.max(rec_las_pixel1.y)))
    yy_range2_ = (max(0, np.min(rec_las_pixel2.y)), min(RSImg2.height, np.max(rec_las_pixel2.y)))
    xx_range1_ = (max(0, np.min(rec_las_pixel1.x)), min(RSImg1.width, np.max(rec_las_pixel1.x)))
    xx_range2_ = (max(0, np.min(rec_las_pixel2.x)), min(RSImg2.width, np.max(rec_las_pixel2.x)))
    img_las_w1, img_las_h1 = np.ceil(xx_range1_[1]) - np.floor(xx_range1_[0]), np.ceil(yy_range1_[1]) - np.floor(yy_range1_[0])
    img_las_w2, img_las_h2 = np.ceil(xx_range2_[1]) - np.floor(xx_range2_[0]), np.ceil(yy_range2_[1]) - np.floor(yy_range2_[0])
    scale_x, scale_y = (xx_range1_[1]-xx_range1_[0])/(xx_range2_[1]-xx_range2_[0]), (yy_range1_[1]-yy_range1_[0])/(yy_range2_[1]-yy_range2_[0])


    # 步骤7 计算交会区域的平均高程以及最大最小高程, 定义高程分层范围和分层数量
    print("="*10 + str(7) + "="*10 + "\n计算高程范围以及平均高程,定义高程分层")
    yy_step1 = 13000
    alt_m, alt_M, alt_e = 170, 230, 200

    alt_offset = 20
    alt_interval_num = 20
    alt_levels = np.linspace(alt_m - alt_offset, alt_M + alt_offset, alt_interval_num)

    # 步骤8 生成核线方程
    print("="*10 + str(8) + "="*10 + "\n核线方程生成")
    epipolar_coeffs1, epipolar_coeffs2 = get_epipolar_coeffs_tonggui(xx_range1_, yy_range1_, yy_step1, RSImg1.rpc_model, RSImg2.rpc_model, alt_levels, alt_e, RSImg1, RSImg2)

    # 步骤9 根据核线方程进行核线影像采样
    print("="*10 + str(9) + "="*10 + "\n核线影像L重采样与保存")
    epipolar_img1, xy_rect1 = resample_epipolar_imageL_tonggui(RSImg1.img, epipolar_coeffs2, xx_range1_, yy_range1_, yy_step1)

    epipolar_img1 = norm_save_epipolar_image(epipolar_img1, save_path + "/eimgL.tif")
    norm_save_epipolar_image(np.transpose(epipolar_img1), save_path + "/eimgL_transp.tif")

    # 步骤10  las1根据核线关系重投影至L
    print("="*10 + str(10) + "="*10 + "\nlas1根据核线关系重投影至核线L")
    rep_las_pixel1 = las_reproject1L(rec_las_pixel1, xy_rect1, xx_range1_, yy_range1_, yy_step1)
    point_2_dsm(rep_las_pixel1, save_path + "/dsm_1_epireproj.tif", xx_range1_, yy_range1_)
    del xy_rect1, epipolar_coeffs2, rec_las_pixel1
    gc.collect()

    # 步骤11 根据核线方程进行核线影像采样
    print("="*10 + str(11) + "="*10 + "\n核线影像R重采样与保存")
    epipolar_img2, xy_rect2 = resample_epipolar_imageR_tonggui(RSImg2.img, epipolar_coeffs1, xx_range1_, yy_range1_, yy_range2_, yy_step1)

    epipolar_img2 = norm_save_epipolar_image(epipolar_img2, save_path + "/eimgR.tif")
    # norm_save_epipolar_image(np.transpose(epipolar_img2), save_path + "/eimgR_transp.tif")

    # 步骤12  las2根据核线关系重投影至R
    print("="*10 + str(12) + "="*10 + "\nlas2根据核线关系重投影至核线R")
    rep_las_pixel2 = las_reproject1R(rec_las_pixel2, xy_rect2, xx_range1_, yy_range1_, xx_range2_, yy_range2_, yy_step1)


    # # xy range 计算更新
    yy_range2_ = (yy_range2_[0], yy_range2_[0] + (yy_range2_[1]-yy_range2_[0]) * scale_y)
    xx_range2_ = (xx_range2_[0], xx_range2_[0] + (xx_range2_[1]-xx_range2_[0]) * scale_x)
    _yy_range2_ = (max(0, np.min(rep_las_pixel2.y)), min(RSImg2.height, np.max(rep_las_pixel2.y)))
    _xx_range2_ = (max(0, np.min(rep_las_pixel2.x)), min(RSImg2.width, np.max(rep_las_pixel2.x)))

    point_2_dsm(rep_las_pixel2, save_path + "/dsm_2_epireproj.tif", xx_range2_, yy_range2_)
    del xy_rect2, epipolar_coeffs1, rec_las_pixel2
    gc.collect()


    # # # 步骤13 R核线数据仿射变换
    print("="*10 + str(13) + "="*10 + "\n核线R影像仿射变换")
    good_matches, good_1points, good_2points = get_good_sifts(epipolar_img1, epipolar_img2, split_nums=22, check_epi=True, epi_axis="W", epi_t=5)
    # lr_xy = [[good_lpoints[m[0].queryIdx].pt[0], good_lpoints[m[0].queryIdx].pt[1], good_rpoints[m[0].trainIdx].pt[0], good_rpoints[m[0].trainIdx].pt[1]] for m in good_matches]
    lr_xy = [[good_2points[m[0].trainIdx].pt[0], good_2points[m[0].trainIdx].pt[1], good_1points[m[0].queryIdx].pt[0], good_1points[m[0].queryIdx].pt[1]] for m in good_matches]
    affine_matrix_2, _ = compute_affine_matrix(lr_xy)
    affine_matrix_2_ = np.linalg.inv(affine_matrix_2)
    epipolar_img2 = img1_apply_transform(epipolar_img2, affine_matrix_2_)
    norm_save_epipolar_image(epipolar_img2, save_path + "/eimgR_transf.tif")
    norm_save_epipolar_image(np.transpose(epipolar_img2), save_path + "/eimgR_transp_transf.tif")

    del epipolar_img2, epipolar_img1, RSImg1
    gc.collect()


    # # # #步骤14 点云仿射变换
    print("="*10 + str(14) + "="*10 + "\nlas2同样仿射变换")
    rep_las_pixel2.x -= xx_range2_[0]
    rep_las_pixel2.y -= yy_range2_[0]
    tranf_rep_las_pixel2_ = points_apply_transform(rep_las_pixel2, affine_matrix_2)
    tranf_rep_las_pixel2_.x += xx_range2_[0]
    tranf_rep_las_pixel2_.y += yy_range2_[0]

    point_2_dsm(tranf_rep_las_pixel2_, save_path+"/dsm2_epireproj_transf.tif", xx_range2_, yy_range2_)

    # 步骤16 获取最终视差与保存
    print("="*10 + str(15) + "="*10 + "\n视差获取与保存")
    disp_res, disp_num = get_disp_res_tonggui3(rep_las_pixel1, tranf_rep_las_pixel2_, yy_range1_, xx_range1_, yy_range2_, xx_range2_)

    # cv2.imwrite(save_path + "/epiDisp.tif", disp_res)
    cv2.imwrite(save_path + "/epiDisp_transp.tif", np.transpose(disp_res))






if __name__ == '__main__':

    # # #  cook
    names = [str(i) for i in range(11, 24)]
    point_paths = ["../cook-las5/filter/1/{}-f.las".format(name) for name in names]
    imgL_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-BWDPAN.tiff"
    rpcL_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-BWDPAN_RPC.txt"
    xmlL_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-BWDPAN.xml"
    imgR_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-FWDPAN.tiff"
    rpcR_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-FWDPAN_RPC.txt"
    xmlR_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-FWDPAN.xml"
    save_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_1"
    main_stereo_tonggui(point_paths, imgL_path, rpcL_path, xmlL_path, imgR_path, rpcR_path, xmlR_path, save_path)


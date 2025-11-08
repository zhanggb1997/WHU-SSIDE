'''
Project    : RSDeploy
FileName   : reprojection_erreo .py
CreateTime : 2025/3/10 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import cv2
import numpy as np
import rasterio
import srtm4
from scipy.optimize import least_squares

from epipolar_check import getSift, getMatch, NNDR, RANSAC
from get_intersection_region import get_inter, get_split_inter_stereo, get_inter_stereo
from image.img_info import RSImage


def slip_inter_geo_stereo(RSImg1, RSImg2, inter_geo_bbox, step=0.01):
    sub_geo_bboxs = []
    sub_pixel_bboxs1 = []
    sub_pixel_bboxs2 = []
    sub_imgs1 = {}
    sub_imgs2 = {}

    inter_geo_bbox = np.array(inter_geo_bbox)
    geo_min_x = min(inter_geo_bbox[:, 0])
    geo_min_y = min(inter_geo_bbox[:, 1])
    geo_max_x = max(inter_geo_bbox[:, 0])
    geo_max_y = max(inter_geo_bbox[:, 1])

    geo_xs = np.arange(geo_min_x, geo_max_x, step)
    geo_ys = np.arange(geo_min_y, geo_max_y, step)

    for i in range(len(geo_xs) - 1):
        for j in range(len(geo_ys) - 1):
            sub_geo_bboxs.append([[geo_xs[i], geo_ys[j]], [geo_xs[i], geo_ys[j+1]], [geo_xs[i+1], geo_ys[j+1]], [geo_xs[i+1], geo_ys[j]]])

    for sub_geo_bbox in sub_geo_bboxs:
        sub_pixel_bbox1 = []
        sub_pixel_bbox2 = []
        for lon, lat in sub_geo_bbox:
            z = srtm4.srtm4(lon, lat)
            x1, y1 = RSImg1.rpc_model.projection(lon, lat, z)
            x2, y2 = RSImg2.rpc_model.projection(lon, lat, z)

            x1 = max(0, min(x1, RSImg1.width))
            y1 = max(0, min(y1, RSImg1.height))
            x2 = max(0, min(x2, RSImg2.width))
            y2 = max(0, min(y2, RSImg2.height))

            sub_pixel_bbox1.append([x1, y1])
            sub_pixel_bbox2.append([x2, y2])

        sub_pixel_bbox1 = np.array(sub_pixel_bbox1)
        sub_pixel_bbox2 = np.array(sub_pixel_bbox2)

        start_x1 = min(sub_pixel_bbox1[:, 0])
        width_x1 = max(sub_pixel_bbox1[:, 0]) - min(sub_pixel_bbox1[:, 0])
        start_y1 = min(sub_pixel_bbox1[:, 1])
        height_y1 = max(sub_pixel_bbox1[:, 1]) - min(sub_pixel_bbox1[:, 1])

        start_x2 = min(sub_pixel_bbox2[:, 0])
        width_x2 = max(sub_pixel_bbox2[:, 0]) - min(sub_pixel_bbox2[:, 0])
        start_y2 = min(sub_pixel_bbox2[:, 1])
        height_y2 = max(sub_pixel_bbox2[:, 1]) - min(sub_pixel_bbox2[:, 1])

        if width_x1 and height_y1 and width_x2 and height_y2:
            sub_pixel_bboxs1.append([start_x1, start_y1, width_x1, height_y1])
            sub_pixel_bboxs2.append([start_x2, start_y2, width_x2, height_y2])


    for sub_pixel_bbox1_, sub_pixel_bbox2_ in zip(sub_pixel_bboxs1, sub_pixel_bboxs2):

        sub_img1 = RSImg1.get_sub_image(sub_pixel_bbox1_)
        sub_img2 = RSImg2.get_sub_image(sub_pixel_bbox2_)

        sub_imgs1[tuple(sub_pixel_bbox1_[:2])] = sub_img1
        sub_imgs2[tuple(sub_pixel_bbox2_[:2])] = sub_img2

    return sub_imgs1, sub_imgs2



def get_sifts(img1_dicts, img2_dicts):
    matches_good_L_ = []
    keyPointL_ = []
    keyPointR_ = []

    for img1_key, img2_key in zip(img1_dicts, img2_dicts):
        x1, y1 = img1_key
        x2, y2 = img2_key
        img1 = img1_dicts[img1_key]
        img2 = img2_dicts[img2_key]
        keyPointL, keyPointR, desL, desR = getSift(img1, img2)

        if desL is None or desR is None:
            continue
        # 进行比对匹配
        MatchesL = getMatch(desL, desR)
        if len(MatchesL[0]) == 1:
            matches_good_L = list(MatchesL)
        else:
            # 比值提纯法
            matches_good_L = NNDR(MatchesL, 0.5, 10)
            # cv2.imwrite("./temp_reproj/NNDR_result.png", cv2.drawMatchesKnn(img1, keyPointL, img2, keyPointR, matches_good_L, None, flags=2))

            # Ransac处理
            matches_good_L, _ = RANSAC(keyPointL, keyPointR, matches_good_L, 2000, 5)
            # cv2.imwrite("./temp_reproj/RANSAC_result.png", cv2.drawMatchesKnn(img1, keyPointL, img2, keyPointR, matches_good_L, None, flags=2))

        # 匹配点信息更新
        pointnumL, pointnumR = len(keyPointL_), len(keyPointR_)
        for temp_match_g in matches_good_L:
            temp_match_g[0].queryIdx += pointnumL
            temp_match_g[0].trainIdx += pointnumR
        matches_good_L_.extend(matches_good_L)

        for pointL in keyPointL:
            pointL.pt = (pointL.pt[0] + x1, pointL.pt[1] + y1)
        for pointR in keyPointR:
            pointR.pt = (pointR.pt[0] + x2, pointR.pt[1] + y2)

        # 添加数据
        keyPointL_.extend(keyPointL)
        keyPointR_.extend(keyPointR)


    return matches_good_L_, keyPointL_, keyPointR_


def save_sifts(keyPoint1, keyPoint2, matches_good_1, save_path):
    print("开始写入匹配点数据")

    x1y1x2y2 = []

    for match in matches_good_1:
        q_id = match[0].queryIdx  # 图1中的查询匹配点id
        p_id = match[0].trainIdx  # 图2中的对应匹配点id
        keyP1 = keyPoint1[q_id]  # 图1中的关键点
        keyP2 = keyPoint2[p_id]  # 图2中的关键点
        x_1, y_1 = keyP1.pt[0], keyP1.pt[1]  # 图1关键点位置 x,y 像素坐标
        x_2, y_2 = keyP2.pt[0], keyP2.pt[1]  # 图2关键点位置 x,y 像素坐标

        # 添加相关信息
        x1y1x2y2.append([x_1, y_1, x_2, y_2])

    with open(save_path, 'w') as file:
        for xyxy in x1y1x2y2:
            # 将每行的4个数值转换为字符串并用空格分隔
            line = '\t'.join(map(str, xyxy))
            file.write(line + '\n')

    print(f"数据已成功写入{save_path}")




def reproject_check(Img1, Img2, sift_path, res_path, DSM_path=None, SEG_path=None):
    rpcm1 = Img1.rpc_model
    rpcm2 = Img2.rpc_model

    points = []

    x1_errors = []
    y1_errors = []
    x2_errors = []
    y2_errors = []

    with open(sift_path, 'r') as file:
        for line in file:
            # 去除行末换行符并按空格分割字符串
            rows_cols = line.strip().split('\t')
            # 将字符串转换为浮点数（或整数）
            row_col = [float(num) for num in rows_cols]
            points.append(row_col)

    # # DSM验证精度
    dsm = None
    if DSM_path:
        dsm = rasterio.open(DSM_path)
        # 步骤1：坐标系验证
        if str(dsm.crs) != 'EPSG:4326':
            raise ValueError("输入影像非WGS84地理坐标系")
    # # 语义排除非地面点
    seg = None
    if SEG_path:
        seg = rasterio.open(SEG_path)
        # 步骤1：坐标系验证
        if str(seg.crs) != 'EPSG:4326':
            raise ValueError("输入影像非WGS84地理坐标系")

    xyxy12 = []
    use_num = []
    for pts in points:
        x_1, y_1, x_2, y_2 = pts  # 关键点位置 x,y 像素坐标

        # 初始化猜测高度
        h0 = 0.0
        try:
            lon0, lat0 = Img1.rpc_model.localization(x_1, y_1, h0)
        except Exception as e:
            use_num.append(False)
            continue
        # 定义误差函数
        def error_func(params):
            lat, lon, h = params
            x1_pred, y1_pred = rpcm1.projection(lon, lat, h)
            x2_pred, y2_pred = rpcm2.projection(lon, lat, h)
            return [x1_pred - x_1, y1_pred - y_1, x2_pred - x_2, y2_pred - y_2]

        # 优化参数
        result = least_squares(
            error_func,
            [lat0, lon0, h0],
            method="lm",
            ftol=1e-6,
            max_nfev=200
        )

        # 输出结果
        lat_opt, lon_opt, h_opt = result.x
        # print(f"物方点坐标：纬度={lat_opt1:.6f}°, 经度={lon_opt1:.6f}°, 高程={h_opt1:.2f}米")

        # DSM高程核验
        if DSM_path:
            row, col = dsm.index(lon_opt, lat_opt)
            # 边界检查
            if not (0 <= row < dsm.height and 0 <= col < dsm.width):
                use_num.append(False)
                # print("DSM边界外！")
                continue
            # 读取像素值（自动处理浮点插值）
            window = rasterio.windows.Window(col_off=int(col), row_off=int(row), width=1, height=1)
            h_opt_ = dsm.read(1, window=window)[0][0]
            if (h_opt_ - h_opt) > 1:
                use_num.append(False)
                print("垂直误差过大:{}".format((h_opt_ - h_opt)))
                continue
            elif np.isnan(h_opt_):
                use_num.append(False)
                print("为Nan！")
                continue

        x1_pred_, y1_pred_ = Img1.rpc_model.projection(lon_opt, lat_opt, h_opt)
        x2_pred_, y2_pred_ = Img2.rpc_model.projection(lon_opt, lat_opt, h_opt)

        x1_error = (x1_pred_ - x_1)
        y1_error = (y1_pred_ - y_1)
        x2_error = (x2_pred_ - x_2)
        y2_error = (y2_pred_ - y_2)

        x1_errors.append(x1_error)
        y1_errors.append(y1_error)
        x2_errors.append(x2_error)
        y2_errors.append(y2_error)

        use_num.append(True)
        xyxy12.append([x_1, y_1, x_2, y_2])

    # print("跳过数量为：{}".format(str(skip_num)))


    # 计算误差均值
    T_ = 4
    x1_errors_ = np.abs(np.array(x1_errors))
    y1_errors_ = np.abs(np.array(y1_errors))
    x2_errors_ = np.abs(np.array(x2_errors))
    y2_errors_ = np.abs(np.array(y2_errors))

    errors_all = x1_errors_ + y1_errors_ + x2_errors_ + y2_errors_

    xyxy12_ = np.array(xyxy12)[errors_all < T_]

    x1_errors_ = x1_errors_[errors_all < T_]
    y1_errors_ = y1_errors_[errors_all < T_]
    x2_errors_ = x2_errors_[errors_all < T_]
    y2_errors_ = y2_errors_[errors_all < T_]

    x1_err_mean = np.mean(x1_errors_)
    y1_err_mean = np.mean(y1_errors_)
    x2_err_mean = np.mean(x2_errors_)
    y2_err_mean = np.mean(y2_errors_)

    x1_err_max = np.max(x1_errors_)
    y1_err_max = np.max(y1_errors_)
    x2_err_max = np.max(x2_errors_)
    y2_err_max = np.max(y2_errors_)

    print("*" * 20)
    lines_mean = "x1 error mean :" + str(x1_err_mean) + "\n" + \
                 "y1 error mean :" + str(y1_err_mean) + "\n" + \
                 "x2 error mean :" + str(x2_err_mean) + "\n" + \
                 "y2 error mean :" + str(y2_err_mean)
    print(lines_mean)
    print("" * 20)
    lines_max = "x1 error max :" + str(x1_err_max) + "\n" + \
                "y1 error max :" + str(y1_err_max) + "\n" + \
                "x2 error max :" + str(x2_err_max) + "\n" + \
                "y2 error max :" + str(y2_err_max)
    print(lines_max)
    print("" * 20)
    print("有效匹配数量为：{}".format(str(len(x1_errors))))

    with open(res_path, 'w') as file:

        file.write(lines_mean + '\n')
        file.write(lines_max + '\n')

        file.write("match points number:{}\n".format(str(len(x1_errors))))

        file.write('x1\ty1\tx2\ty2\tx1-error\ty1-error\tx2-error\ty2-error\n')

        # 写入/查看匹配点的误差
        for xy_e_no in range(len(x1_errors_)):
            x1_e_temp, y1_e_temp, x2_e_temp, y2_e_temp = x1_errors_[xy_e_no], y1_errors_[xy_e_no], x2_errors_[xy_e_no], y2_errors_[xy_e_no]
            x1, y1, x2, y2 = xyxy12_[xy_e_no, 0], xyxy12_[xy_e_no][1], xyxy12_[xy_e_no][2], xyxy12_[xy_e_no][3]
            line_error = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(x1, y1, x2, y2, x1_e_temp, y1_e_temp, x2_e_temp, y2_e_temp)
            file.write(line_error + '\n')

            # if 6 > abs(x1_e_temp) + abs(y1_e_temp) + abs(x2_e_temp) + abs(y2_e_temp) > 4:
            #     x1_temp, y1_temp, x2_temp, y2_temp = points[xy_e_no]
            #     Img1.show_points(np.array([[x1_temp, y1_temp]]))
            #     Img2.show_points(np.array([[x2_temp, y2_temp]]))




def reprojection_error(img1_path, rpc1_path, xml1_path, img2_path, rpc2_path, xml2_path, sift_save_path, res_save_path, dsm_path=None):
    print("1 读取影像信息")
    RSImg1 = RSImage(img1_path, rpc1_path, xml1_path)
    RSImg2 = RSImage(img2_path, rpc2_path, xml2_path)

    # print("2 计算影像交会区域")
    # inter_geo_bbox = get_inter(RSImg1, RSImg2)
    #
    # print("3 切分影像数据")
    # sub_img_dicts1, sub_img_dicts2 = slip_inter_geo_stereo(RSImg1, RSImg2, inter_geo_bbox)
    #
    # print("4 影像进行匹配")
    # matches_good_L, keyPointL, keyPointR = get_sifts(sub_img_dicts1, sub_img_dicts2)
    #
    # print("5 保存匹配点信息")
    # save_sifts(keyPointL, keyPointR, matches_good_L, sift_save_path)

    print("6 重投影误差计算")
    reproject_check(RSImg1, RSImg2, sift_save_path, res_save_path, DSM_path=dsm_path)








if __name__ == '__main__':
    dsm_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/las_dsm_all/merged_output_max_wgs84.tif"
    imgL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-BWDPAN.tiff"
    rpcL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-BWDPAN_rpc.txt"
    xmlL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-BWDPAN.xml"
    imgR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-FWDPAN.tiff"
    rpcR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-FWDPAN_rpc.txt"
    xmlR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412-FWDPAN.xml"
    sift_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF07_sifts.txt"
    res_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412/GF07_reprojerros1m.txt"

    # imgR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-FWDPAN.tiff"
    # rpcR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-FWDPAN_ba_RPC1.txt"
    # xmlR_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-FWDPAN.xml"
    # imgL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-BWDPAN.tiff"
    # rpcL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-BWDPAN_ba_RPC1.txt"
    # xmlL_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901-BWDPAN.xml"
    # sift_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF07_sifts2.txt"
    # res_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901/GF07_reproj2.txt"

    # imgL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_PM1_041907_20220521_KS450_01_012_L1A_01_202502180000160001/GF02_PA1_041907_20220521_KS450_01_012_L1A_01.tif"
    # rpcL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_PM1_041907_20220521_KS450_01_012_L1A_01_202502180000160001/GF02_PA1_041907_20220521_KS450_01_012_L1A_01_ba_rpc1.txt"
    # xmlL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_PM1_041907_20220521_KS450_01_012_L1A_01_202502180000160001/GF02_PA1_041907_20220521_KS450_01_012_L1A_01.meta.xml"
    # imgR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_202502170000260001/GF06_PAN_014166_20210117_MY261_01_037_L1A_01.tif"
    # rpcR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_202502170000260001/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_ba_rpc1.txt"
    # xmlR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF06_PAN_014166_20210117_MY261_01_037_L1A_01_202502170000260001/GF06_PAN_014166_20210117_MY261_01_037_L1A_01.meta.xml"
    # sift_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_GF06_sifts.txt"
    # res_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/GF02_GF06_reproj.txt"

    # imgL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-FWDPAN.tiff"
    # rpcL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-FWDPAN_RPC.txt"
    # xmlL_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-FWDPAN.xml"
    # imgR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-BWDPAN.tiff"
    # rpcR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-BWDPAN_RPC.txt"
    # xmlR_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694/GF7_DLC_E113.7_N36.0_20221231_L1A0001166694-BWDPAN.xml"
    # sift_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF07_GF07_sifts.txt"
    # res_save_path = r"/home/dshare/01Data/3DDisp/LiDAR/NJ/images/tempGF7/GF07_GF07_reproj.txt"

    reprojection_error(imgL_path, rpcL_path, xmlL_path, imgR_path, rpcR_path, xmlR_path, sift_save_path, res_save_path, dsm_path)

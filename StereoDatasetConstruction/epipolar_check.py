'''
Project    : RSDeploy
FileName   : epipolar_check .py
CreateTime : 2025/2/21 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import glob
import os
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def getImgPair(imgLPath, imgRPath, dispPath=None, scale=1):
    imgL = cv2.imread(imgLPath, flags=cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgRPath, flags=cv2.IMREAD_GRAYSCALE)
    if dispPath:
        disp = cv2.imread(dispPath, flags=cv2.IMREAD_UNCHANGED)


    if scale == 1:
        pass
    else:
        imgL = cv2.resize(imgL, (round(imgL.shape[1] / scale), round(imgL.shape[0] / scale)))
        imgR = cv2.resize(imgR, (round(imgR.shape[1] / scale), round(imgR.shape[0] / scale)))

    if dispPath:
        return imgL, imgR, disp


    return imgL, imgR


# def getImgPathPair(imgLDir, imgRDir, dispDir):
def getImgPathPair(imgLDir, imgRDir, dispDir=None):
    imgLPaths = natsorted(glob.glob(os.path.join(imgLDir + "/*.tif*")))
    imgRPaths = natsorted(glob.glob(os.path.join(imgRDir + "/*.tif*")))
    # imgLPaths = natsorted(glob.glob(os.path.join(imgLDir + "/*B_F*.tif*")))
    # imgRPaths = natsorted(glob.glob(os.path.join(imgRDir + "/*B_F*.tif*")))

    if dispDir:
        dispPaths = natsorted(glob.glob(os.path.join(dispDir + "/*.tif*")))
        # dispPaths = natsorted(glob.glob(os.path.join(dispDir + "/*B_F*.tif*")))
        return imgLPaths, imgRPaths, dispPaths

    return imgLPaths, imgRPaths

def getSift(imgGrayL, imgGrayR):
    siftFun = cv2.SIFT_create()  # 创建sift
    keyPointL, desL = siftFun.detectAndCompute(imgGrayL, None)  # 计算获得imgL关键点和描述子
    keyPointR, desR = siftFun.detectAndCompute(imgGrayR, None)  # 计算获得imgR关键点和描述子
    # print("Left图像的关键点数量：" + str(len(keyPointL)) + '\n' + "Right图像的关键点数量：" + str(len(keyPointR)))

    return keyPointL, keyPointR, desL, desR


def getMatch(desL, desR):
    # 关键点匹配BFMatcher
    bf = cv2.BFMatcher()

    matchesL = bf.knnMatch(desL, desR, k=2)
    # matchesR = bf.knnMatch(desR, desL, k=2)

    # return matchesL, matchesR
    return matchesL


# 比值提纯法
def NNDR(matches, alpha, min_matches=0):
    # 将要筛选的匹配点
    matches_good = []
    # 开始循环筛选，要满足最少n个匹配点
    if min_matches:
        do_redo = True
        all_time = 0
        while do_redo:
            for m, n in matches:
                if m.distance < alpha * n.distance:
                    matches_good.append([m])
            # 达到要求的话，结束
            if len(matches_good) >= min_matches:
                do_redo = False
            # 没有重新匹配
            else:
                matches_good = []
                alpha += 0.05
                all_time += 1

                if all_time > 10:
                    do_redo = False
    else:  # 没有个数要求的话
        for m, n in matches:
            if m.distance < alpha * n.distance:
                matches_good.append([m])

    return matches_good


def RANSAC(keyPointL, keyPointR, matches, max_iter=1000, T=5.):
    # 转换为坐标数组
    src_pts = np.float32([keyPointL[m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keyPointR[m[0].trainIdx].pt for m in matches]).reshape(-1, 2)

    # # 使用RANSAC计算单应性矩阵
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, T, maxIters=max_iter)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, T, confidence=0.99, maxIters=max_iter)

    # 提取内点
    inlier_matches = [matches[i] for i in range(len(mask)) if mask[i]]

    return inlier_matches, mask


def check_disparity(keyPointL, keyPointR, matches, disparity, epi_T=1.25, NONE_VALUE=-999):
    disp_value_all = np.zeros_like(disparity, dtype=float)  # 影像对应位置匹配计算所得视差总值
    disp_value_mean = np.zeros_like(disparity, dtype=float)  # 影像对应位置匹配计算所得视差平均值
    disp_point_num = np.zeros_like(disparity, dtype=int)  # 影像对应位置匹配点数量

    good_matches_ = []  # 优良的匹配点
    bad_matches_ = []  # 差异大的匹配点
    good_disp_ = []  # 优良匹配点计算所得视差值与真值的差异
    bad_disp_ = []  # 差异大匹配点计算所得视差值与真值的差异

    good_record_dict = {}  # 记录字典
    bad_record_dict = {}  # 记录字典
    disp_err_record = []  # 记录字典

    # 遍历计算差异
    for no, m in enumerate(matches):
        q_id = m[0].queryIdx  # 左视的关键点编号
        p_id = m[0].trainIdx  # 右视的关键点编号
        keyPL = keyPointL[q_id]  # 左视关键点
        keyPR = keyPointR[p_id]  # 右视关键点

        sift_disp = float(keyPL.pt[0] - keyPR.pt[0])  # 左右视关键点横坐标相减
        w_int, h_int = int(keyPL.pt[0]), int(keyPL.pt[1])  # 左视的关键点横纵坐标
        truth_disp = disparity[h_int, w_int]  # 对应的真值视差

        disp_err = sift_disp-truth_disp
        disp_err_record.append(disp_err)

        # 无效视差处，跳过
        if truth_disp <= NONE_VALUE:
            pass
        # 左右匹配点横坐标差值过大，放弃
        elif sift_disp > 350:
            bad_record_dict[(w_int, h_int)] = no
        # err差值过大，放弃
        elif abs(disp_err) > 100:
            bad_record_dict[(w_int, h_int)] = no
        # 左右匹配点的纵向视差不符合核线，跳过
        elif abs(keyPL.pt[1] - keyPR.pt[1]) > epi_T:
            bad_record_dict[(w_int, h_int)] = no
        # 左视上同一个点，在右视图上匹配到了多个点，大于某一阈值进行进一步判断
        elif disp_point_num[h_int, w_int] and abs(sift_disp - disp_value_mean[h_int, w_int]) > 1.0:
            print("H:{}, W:{}, 两匹配点差异为：{}".format(h_int, w_int, abs(sift_disp - disp_value_mean[h_int, w_int])))
            # 如果这个匹配点和真值视差值相近，最开始的判定与真值视差值相差较大，则判定其最开始的判定为错误的，该点作为有效点进行输入，并删除之前的匹配点
            if abs(truth_disp - disp_value_mean[h_int, w_int]) > abs(truth_disp - sift_disp):
                print("当前匹配点作为正确点载入! 并删除以往的错误点！")
                # 删除累加的和，并将该点作为第一个点初始点输入
                disp_value_all[h_int, w_int] = sift_disp
                # 因为是第一个点，所以mean值=all值
                disp_value_mean[h_int, w_int] = sift_disp
                # 点数量为1
                disp_point_num[h_int, w_int] = 1

                bad_record_dict[(w_int, h_int)] = good_record_dict[(w_int, h_int)]
                good_record_dict[(w_int, h_int)] = no
            # 新的匹配点与真实视差值大于老点，放弃作为有效点，跳过
            else:
                pass
        # 左视上同一个点，在右视图上匹配到了多个点，并且小于某一阈值，进行添加
        elif disp_point_num[h_int, w_int]:
            # 作为有效点，disp mean值=all值
            disp_value_all[h_int, w_int] += sift_disp
            # 匹配点数量=1
            disp_point_num[h_int, w_int] += 1
            # 计算均值
            disp_value_mean[h_int, w_int] = disp_value_all[h_int, w_int] / disp_point_num[h_int, w_int]

            good_record_dict[(w_int, h_int)] = (good_record_dict[(w_int, h_int)] if isinstance(good_record_dict[(w_int, h_int)], list) else [good_record_dict[(w_int, h_int)]]) + [no, ]
        else:
            # 确保是第一个有效点
            assert (disp_value_all[h_int, w_int] == 0) and (disp_value_mean[h_int, w_int] == 0) and (disp_point_num[h_int, w_int] == 0)
            # 作为第一个有效点，disp总和累加
            disp_value_all[h_int, w_int] = sift_disp
            # 匹配点数量1
            disp_point_num[h_int, w_int] = 1
            # 均值
            disp_value_mean[h_int, w_int] = disp_value_all[h_int, w_int]

            good_record_dict[(w_int, h_int)] = no


    # 转化输出结果
    for good_record_key in good_record_dict.keys():
        good_record_value = good_record_dict[good_record_key]
        if isinstance(good_record_value, list):
            for v_no in good_record_value:
                good_matches_.append(matches[v_no])
                good_disp_.append(disp_err_record[v_no])
        else:
            good_matches_.append(matches[good_record_value])
            good_disp_.append(disp_err_record[good_record_value])

    for bad_record_key in bad_record_dict.keys():
        bad_record_value = bad_record_dict[bad_record_key]
        if isinstance(bad_record_value, list):
            for v_no in bad_record_value:
                bad_matches_.append(matches[v_no])
                bad_disp_.append(disp_err_record[v_no])
        else:
            bad_matches_.append(matches[bad_record_value])
            bad_disp_.append(disp_err_record[bad_record_value])

    return good_disp_, bad_disp_, good_matches_, bad_matches_



def check_epipolar(keyPointL, keyPointR, matches, axis="H", T_=15.0):

    good_epipolars_ = []
    bad_epipolars_ = []
    good_matches_ = []
    bad_matches_ = []

    # 遍历匹配点
    for m in matches:
        q_id = m[0].queryIdx  # 左图中的查询匹配点id
        p_id = m[0].trainIdx  # 右图中的对应匹配点id
        keyPL = keyPointL[q_id]  # 左图中的关键点
        keyPR = keyPointR[p_id]  # 右图中的关键点

        x_l, y_l = keyPL.pt[0], keyPL.pt[1]  # 左图关键点位置 x,y 坐标
        x_r, y_r = keyPR.pt[0], keyPR.pt[1]  # 右图关键点位置 x,y 坐标

        if axis=="H":
            epipolar_disp = y_l - y_r # 左右图关键点y坐标计算核线y轴差值
        elif axis=="W":
            epipolar_disp = x_l - x_r # 左右图关键点y坐标计算核线y轴差值
        else:
            return

        if abs(epipolar_disp) < T_:
            good_matches_.append(m)
            good_epipolars_.append(epipolar_disp)
        else:
            bad_matches_.append(m)
            bad_epipolars_.append(epipolar_disp)

    good_epipolar_ = np.array(good_epipolars_)
    bad_epipolar_ = np.array(bad_epipolars_)


    return good_epipolar_, bad_epipolar_, good_matches_, bad_matches_


def drawMatch(imgL, keyPointL, imgR, keyPointR, matches, epi_disps, basename):
    # 视差数值
    for no, m in enumerate(matches):
        q_id = m[0].queryIdx
        p_id = m[0].trainIdx
        keyPL = keyPointL[q_id]
        keyPR = keyPointR[p_id]
        x_int, y_int = int(keyPL.pt[0]), int(keyPL.pt[1])
        epi_disp = epi_disps[no]  # 对应的真值视差
        imgL = cv2.putText(imgL, "%.3f"%epi_disp, (x_int, y_int), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    # 绘制匹配
    drawMatches = cv2.drawMatchesKnn(imgL, keyPointL, imgR, keyPointR, matches, None, flags=2)
    cv2.imwrite("./temp/" + basename, drawMatches)

def drawMatchError(imgL, keyPointL, imgR, keyPointR, matches, epi_disps, basename):
    # 视差数值
    for no, m in enumerate(matches):
        q_id = m[0].queryIdx
        p_id = m[0].trainIdx
        keyPL = keyPointL[q_id]
        keyPR = keyPointR[p_id]
        x_intL, y_intL = keyPL.pt[0], keyPL.pt[1]
        x_intR, y_intR = keyPR.pt[0], keyPR.pt[1]
        epi_error = abs(epi_disps[no])  # 对应的真值视差
        if epi_error > 2.0:
            imgL = cv2.putText(imgL, "%.3f"%epi_error, (int(x_intL), int(y_intR)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        else:
            imgL = cv2.putText(imgL, "%.3f" % epi_error, (int(x_intL), int(y_intR)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    # 绘制匹配
    drawMatches = cv2.drawMatchesKnn(imgL, keyPointL, imgR, keyPointR, matches, None, flags=2)
    cv2.imwrite("./temp/" + basename, drawMatches)


def get_good_sifts(imgL, imgR, split_nums=8, check_epi=True, epi_axis='H', epi_t=5):
    # 整个图的信息
    Good_Matches = []
    Good_Lpoints = []
    Good_Rpoints = []

    heightL, widthL = imgL.shape[:2]
    heightR, widthR = imgR.shape[:2]

    if not (heightL == heightR and widthL == widthR):
        print("heightL{} heightR{} widthL{} widthR{} 不相等".format(heightL, heightR, widthL, widthR))

    x_range = np.linspace(0, widthL, split_nums).astype(int)
    y_range = np.linspace(0, heightL, split_nums).astype(int)

    for y_no, y_ in enumerate(y_range[:-1]):
        for x_no, x_ in enumerate(x_range[:-1]):
            imgL_ = imgL[y_: y_range[y_no + 1], x_: x_range[x_no + 1]]
            imgR_ = imgR[y_: y_range[y_no + 1], x_: x_range[x_no + 1]]

            # 获取对应匹配点
            keyPointL_, keyPointR_, desL_, desR_ = getSift(imgL_, imgR_)
            if desL_ is None or desR_ is None:
                continue

            # 进行比对匹配
            MatchesL_ = getMatch(desL_, desR_)
            if len(MatchesL_[0]) == 1:
                matches_good_L_1 = list(MatchesL_)
            else:
                # 比值提纯法
                matches_good_L_1 = NNDR(MatchesL_, 0.5, 30)
                # cv2.imwrite("./temp/NNDR_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_1, None, flags=2))

            # 检查上下核线
            if check_epi:
                _, _, matches_good_L_2, _ = check_epipolar(keyPointL_, keyPointR_, matches_good_L_1, epi_axis, epi_t*2)
                # cv2.imwrite("./temp/EPI_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_2, None, flags=2))
            else:
                matches_good_L_2 = matches_good_L_1

            # Ransac处理
            if len(matches_good_L_2) > 7:
                matches_good_L_3, mask = RANSAC(keyPointL_, keyPointR_, matches_good_L_2, 1000, 0.8)
                # cv2.imwrite("./temp/RANSAC_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_3, None, flags=2))
            else:
                matches_good_L_3 = matches_good_L_2

            # 再次检查上下核线
            if check_epi:
                _, _, matches_good_L_4, _ = check_epipolar(keyPointL_, keyPointR_, matches_good_L_3, epi_axis, epi_t)
                # cv2.imwrite("./temp/Re_EPI_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_4, None, flags=2))
            else:
                matches_good_L_4 = matches_good_L_3

            # 匹配点信息更新
            pointnumL, pointnumR = len(Good_Lpoints), len(Good_Rpoints)
            for temp_match_g in matches_good_L_4:
                temp_match_g[0].queryIdx += pointnumL
                temp_match_g[0].trainIdx += pointnumR
            Good_Matches.extend(matches_good_L_4)

            for pointL in keyPointL_:
                pointL.pt = (pointL.pt[0] + x_, pointL.pt[1] + y_)
            for pointR in keyPointR_:
                pointR.pt = (pointR.pt[0] + x_, pointR.pt[1] + y_)

            # 添加数据
            Good_Lpoints.extend(keyPointL_)
            Good_Rpoints.extend(keyPointR_)

    # 整体Ransac处理
    if len(Good_Matches) > 7:
        Good_Matches_, mask = RANSAC(Good_Lpoints, Good_Rpoints, Good_Matches, 10000, 0.8)
        # cv2.imwrite("./temp/All_RANSAC_result.png", cv2.drawMatchesKnn(imgL, Good_Lpoints, imgR, Good_Rpoints, Good_Matches, None, flags=2))
        Good_Lpoints_, Good_Rpoints_ = Good_Lpoints, Good_Rpoints
    else:
        Good_Matches_, Good_Lpoints_, Good_Rpoints_ = Good_Matches, Good_Lpoints, Good_Rpoints

    return Good_Matches_, Good_Lpoints_, Good_Rpoints_


def sift_check_epipolar(imgL_path, imgR_path, disp_path=None, split_nums=8, save_path=None):
    if disp_path:
        imgLPaths, imgRPaths, dispPaths = getImgPathPair(imgL_path, imgR_path, disp_path)
        assert len(imgLPaths) == len(imgRPaths) == len(dispPaths)
    else:
        imgLPaths, imgRPaths = getImgPathPair(imgL_path, imgR_path)
        assert len(imgLPaths) == len(imgRPaths)

    # 总核线误差
    epip_err_all = []
    disp_err_all = []

    for i in tqdm(range(len(imgLPaths))):
        print("\n" + "*" * 30)
        print("正在处理：{}".format(os.path.basename(imgLPaths[i])))

        # 读取图像对
        if disp_path:
            imgL, imgR, disp = getImgPair(imgLPaths[i], imgRPaths[i], dispPaths[i], 1)
        else:
            imgL, imgR = getImgPair(imgLPaths[i], imgRPaths[i], None, 1)

        # disp = disp / 256
        # disp[disp==0] = -999

        # 获取优良匹配点
        good_matches, good_lpoints, good_rpoints = get_good_sifts(imgL, imgR, split_nums, True, epi_t=3)

        # 计算核线误差
        good_epip, _, good_epip_matches, _ = check_epipolar(good_lpoints, good_rpoints, good_matches, "H", 2.0)

        # # # 显示匹配点及核线误差
        # drawMatch(imgL, good_lpoints, imgR, good_rpoints, good_epip_matches, good_epip, os.path.splitext(os.path.basename(imgLPaths[i]))[0]+".png")

        # 核线误差更新
        epip_err_all.extend(good_epip)

        # 检测结果计算
        epi_pot_num = len(good_epip)
        if epi_pot_num > 0:
            epi_err_mean = np.mean(np.abs(good_epip))
            epi_err_sqrt = np.sqrt(np.mean(np.abs(good_epip) ** 2))
            epi_err_max  = np.max(np.abs(good_epip))
            epi_err_min  = np.min(np.abs(good_epip))
            epi_err_sum  = np.sum(good_epip) / len(good_epip)

            # 输出核线检查结果
            print("=" * 30)
            print("核线检验")
            print("有效匹配点数量为：" + str(len(good_epip)))
            print("核线平均差值为：" + str(np.mean(np.abs(good_epip))))
            print("核线RMSE差值为：" + str(np.sqrt(np.mean(np.abs(good_epip) ** 2))))
            print("核线差值平均和为：" + str((np.sum(good_epip)) / len(good_epip)))
            print("核线最大差值为：" + str(np.max(np.abs(good_epip))))
            print("核线最小差值为：" + str(np.min(np.abs(good_epip))))
        else:
            epi_err_mean = 0
            epi_err_sqrt = 0
            epi_err_max = 0
            epi_err_min = 0
            epi_err_sum = 0

            # 输出核线检查结果
            print("=" * 30)
            print("核线检验")
            print("有效匹配点数量为：" + str(0))
            print("核线平均差值为：" + str(0))
            print("核线RMSE差值为：" + str(0))
            print("核线差值平均和为：" + str(0))
            print("核线最大差值为：" + str(0))
            print("核线最小差值为：" + str(0))

        # 写入内容
        if disp_path:
            epi_content_ = os.path.basename(imgLPaths[i]) + " " + os.path.basename(imgRPaths[i]) + " " + os.path.basename(dispPaths[i]) + " {} {:.4f} {:.4f} {:.4f} {} {} ".format(epi_pot_num, epi_err_mean, epi_err_sqrt, epi_err_sum, epi_err_max, epi_err_min)
        else:
            epi_content_ = os.path.basename(imgLPaths[i]) + " " + os.path.basename(imgRPaths[i]) + " {} {:.4f} {:.4f} {:.4f} {} {} ".format(epi_pot_num, epi_err_mean, epi_err_sqrt, epi_err_sum, epi_err_max, epi_err_min)
        with open(save_path, 'a+') as f:
            f.write(epi_content_)


        # 计算视差误差
        if disp_path:
            good_disp, bad_disp, good_disp_matches, bad_disp_matches = check_disparity(good_lpoints, good_rpoints, good_matches, disp, 2, -999)
            # 显示匹配点及视差误差
            # drawMatch(imgL, good_lpoints, imgR, good_rpoints, good_disp_matches, good_disp, "GoodDisp_"+os.path.splitext(os.path.basename(imgLPaths[i]))[0]+".png")
            # drawMatchError(imgL, good_lpoints, imgR, good_rpoints, good_disp_matches, good_disp, "CheckDisp_"+os.path.splitext(os.path.basename(imgLPaths[i]))[0]+".jpg")
            disp_err_all.extend(good_disp)

            if len(good_disp) > 0:
                disp_pot_num = len(good_disp)
                disp_err_mean = np.mean(np.abs(good_disp))
                disp_err_sqrt = np.sqrt(np.mean(np.abs(good_epip) ** 2))
                disp_err_max = np.max(np.abs(good_disp))
                disp_err_min = np.min(np.abs(good_disp))
                disp_err_sum = np.sum(good_disp) / len(good_disp)
                # 输出视差检查结果
                print("=" * 30)
                print("视差检验")
                print("有效匹配点数量为：" + str(len(good_disp)))
                print("视差平均差值为：" + str(np.mean(np.abs(good_disp))))
                print("视差RMSE差值为：" + str(np.sqrt(np.mean(np.abs(good_epip) ** 2))))
                print("视差最大差值为：" + str(np.max(np.abs(good_disp))))
                print("视差最小差值为：" + str(np.min(np.abs(good_disp))))
                print("视差差值平均和为：" + str((np.sum(good_disp)) / len(good_disp)))
            else:
                disp_pot_num = 0
                disp_err_mean = 0
                disp_err_sqrt = 0
                disp_err_max = 0
                disp_err_min = 0
                disp_err_sum = 0
                # 输出视差检查结果
                print("=" * 30)
                print("视差检验")
                print("有效匹配点数量为：" + str(0))
                print("视差平均差值为：" + str(0))
                print("视差RMSE差值为：" + str(0))
                print("视差最大差值为：" + str(0))
                print("视差最小差值为：" + str(0))
                print("视差差值平均和为：" + str(0))

            # 写入内容
            disp_content_ = "{} {:.4f} {:.4f} {:.4f} {} {}".format(disp_pot_num, disp_err_mean, disp_err_sqrt, disp_err_sum, disp_err_max, disp_err_min)
            with open(save_path, 'a+') as f:
                f.write(disp_content_)

        with open(save_path, 'a+') as f:
            f.write('\n')


    # 最终计算全部数据
    epipolar_err_all = np.array(list(epip_err_all))
    print("\n" + "="*30)
    print("*"*30)
    print("有效匹配点数量为：" + str(len(epipolar_err_all)))
    print("核线平均差值为：" + str(np.mean(np.abs(epipolar_err_all))))
    print("核线RMSE差值为：" + str(np.sqrt(np.mean(np.abs(epipolar_err_all) ** 2))))
    print("核线最大差值为：" + str(np.max(np.abs(epipolar_err_all))))
    print("核线最小差值为：" + str(np.min(np.abs(epipolar_err_all))))
    print("核线差值平均和为：" + str((np.sum(epipolar_err_all))/len(epipolar_err_all)))

    with open(save_path, 'a+') as f:
        f.write('*' * 60 + '\n')
        f.write("{} {:.4f} {:.4f} {:.4f} {} {} ".format(len(epipolar_err_all), np.mean(np.abs(epipolar_err_all)), np.sqrt(np.mean(np.abs(epipolar_err_all) ** 2)),
                                                       np.max(np.abs(epipolar_err_all)), np.min(np.abs(epipolar_err_all)), (np.sum(epipolar_err_all))/len(epipolar_err_all)))


    if disp_path:
        disparity_err_all = np.array(list(disp_err_all))
        print("*" * 30)
        # print(disp_err_all)
        print("有效匹配点数量为：" + str(len(disparity_err_all)))
        print("视差平均差值为：" + str(np.mean(np.abs(disparity_err_all))))
        print("视差RMSE差值为：" + str(np.sqrt(np.mean(np.abs(disparity_err_all) ** 2))))
        print("视差最大差值为：" + str(np.max(np.abs(disparity_err_all))))
        print("视差最小差值为：" + str(np.min(np.abs(disparity_err_all))))
        print("视差差值平均和为：" + str((np.sum(disparity_err_all)) / len(disparity_err_all)))
        with open(save_path, 'a+') as f:
            f.write("{} {:.4f} {:.4f} {:.4f} {} {}".format(len(disparity_err_all), np.mean(np.abs(disparity_err_all)),
                                                           np.sqrt(np.mean(np.abs(disparity_err_all) ** 2)),
                                                           np.max(np.abs(disparity_err_all)),
                                                           np.min(np.abs(disparity_err_all)),
                                                           (np.sum(disparity_err_all)) / len(disparity_err_all)))


#
# def sift_check_epipolar(imgL_path, imgR_path, disp_path=None, split_nums=8):
#     if disp_path:
#         imgLPaths, imgRPaths, dispPaths = getImgPathPair(imgL_path, imgR_path, disp_path)
#         assert len(imgLPaths) == len(imgRPaths) == len(dispPaths)
#     else:
#         imgLPaths, imgRPaths = getImgPathPair(imgL_path, imgR_path)
#         assert len(imgLPaths) == len(imgRPaths)
#
#     epipolar_disps_all = []
#
#     for i in range(len(imgLPaths)):
#         print("\n" + "*" * 30)
#         print("正在处理：{}".format(os.path.basename(imgLPaths[i])))
#         # 读取图像对
#         imgL, imgR = getImgPair(imgLPaths[i], imgRPaths[i], None, 1)
#
#         heightL, widthL = imgL.shape[:2]
#         heightR, widthR = imgR.shape[:2]
#
#         if not (heightL == heightR and widthL == widthR):
#             print("heightL{} heightR{} widthL{} widthR{} 不相等".format(heightL, heightR, widthL, widthR))
#
#         # 整个图的信息
#         keyPointL, keyPointR = [], []
#         matches_good_L = []
#         matches_bad_L = []
#         good_epipolar_disps = []
#         bad_epipolar_disps = []
#
#         x_range = np.linspace(0, widthL, split_nums).astype(int)
#         y_range = np.linspace(0, heightL, split_nums).astype(int)
#
#         for y_no, y_ in enumerate(y_range[:-1]):
#             for x_no, x_ in enumerate(x_range[:-1]):
#                 imgL_ = imgL[y_: y_range[y_no + 1], x_: x_range[x_no + 1]]
#                 imgR_ = imgR[y_: y_range[y_no + 1], x_: x_range[x_no + 1]]
#                 # 获取对应匹配点
#                 keyPointL_, keyPointR_, desL_, desR_ = getSift(imgL_, imgR_)
#                 if desL_ is None or desR_ is None:
#                     continue
#
#                 # 进行比对匹配
#                 MatchesL_ = getMatch(desL_, desR_)
#                 if len(MatchesL_[0]) == 1:
#                     matches_good_L_1 = list(MatchesL_)
#                 else:
#                     # 比值提纯法
#                     matches_good_L_1 = NNDR(MatchesL_, 0.5, 30)
#                     cv2.imwrite("./temp/NNDR_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_1, None, flags=2))
#
#                 # 检查上下核线
#                 good_epipolar_disps_, bad_epipolar_disps_, matches_good_L_2, matches_bad_L_2 = check_epipolar(keyPointL_, keyPointR_, matches_good_L_1)
#                 cv2.imwrite("./temp/EPI_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_2, None, flags=2))
#
#                 # Ransac处理
#                 if len(matches_good_L_2) > 4:
#                     matches_good_L_3, mask = RANSAC(keyPointL_, keyPointR_, matches_good_L_2, 1000, 1.)
#                     cv2.imwrite("./temp/RANSAC_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_3, None, flags=2))
#                     good_epipolar_disps_ = np.array([good_epipolar_disps_[m_no] for m_no in range(len(mask)) if mask[m_no]])
#                 else:
#                     matches_good_L_3 = matches_good_L_2
#
#                 # 再次检查上下核线
#                 good_epipolar_disps_, bad_epipolar_disps_, matches_good_L_4, matches_bad_L_4 = check_epipolar(keyPointL_, keyPointR_, matches_good_L_3, 3)
#                 cv2.imwrite("./temp/Re_EPI_result.png", cv2.drawMatchesKnn(imgL_, keyPointL_, imgR_, keyPointR_, matches_good_L_4, None, flags=2))
#
#                 # 匹配点信息更新
#                 pointnumL, pointnumR = len(keyPointL), len(keyPointR)
#                 for temp_match_g in matches_good_L_4:
#                     temp_match_g[0].queryIdx += pointnumL
#                     temp_match_g[0].trainIdx += pointnumR
#                 matches_good_L.extend(matches_good_L_4)
#                 # for temp_match_b in matches_bad_L_2:
#                 #     temp_match_b[0].queryIdx += pointnumL
#                 #     temp_match_b[0].trainIdx += pointnumR
#                 # matches_bad_L.extend(matches_bad_L_2)
#
#                 for pointL in keyPointL_:
#                     pointL.pt = (pointL.pt[0] + x_, pointL.pt[1] + y_)
#                 for pointR in keyPointR_:
#                     pointR.pt = (pointR.pt[0] + x_, pointR.pt[1] + y_)
#
#                 # 添加数据
#                 keyPointL.extend(keyPointL_)
#                 keyPointR.extend(keyPointR_)
#                 good_epipolar_disps.extend(good_epipolar_disps_)
#                 bad_epipolar_disps.extend(bad_epipolar_disps_)
#
#         epipolar_disps_all.extend(good_epipolar_disps)
#
#         # # 显示匹配点
#         # drawMatch(imgL, keyPointL, imgR, keyPointR, matches_bad_L, bad_epipolar_disps, "epipolar_matches{}.png".format(str(i)))
#         drawMatch(imgL, keyPointL, imgR, keyPointR, matches_good_L, good_epipolar_disps, "epipolar_matches{}.png".format(str(i)))
#
#         # 输出匹配结果
#         print("有效匹配点数量为：" + str(len(good_epipolar_disps)))
#         print("核线平均差值为：" + str(np.mean(np.abs(good_epipolar_disps))))
#         print("核线RMSE差值为：" + str(np.sqrt(np.mean(good_epipolar_disps) ** 2)))
#         print("核线最大差值为：" + str(np.max(np.abs(good_epipolar_disps))))
#         print("核线最小差值为：" + str(np.min(np.abs(good_epipolar_disps))))
#         print("核线差值平均和为：" + str((np.sum(good_epipolar_disps)) / len(good_epipolar_disps)))
#
#     # 最终计算全部数据
#     epipolar_disps_all = np.array(list(epipolar_disps_all))
#     print("\n" + "="*30)
#     print("有效匹配点数量为：" + str(len(epipolar_disps_all)))
#     print("核线平均差值为：" + str(np.mean(np.abs(epipolar_disps_all))))
#     print("核线RMSE差值为：" + str(np.sqrt(np.mean(epipolar_disps_all) ** 2)))
#     print("核线最大差值为：" + str(np.max(np.abs(epipolar_disps_all))))
#     print("核线最小差值为：" + str(np.min(np.abs(epipolar_disps_all))))
#     print("核线差值平均和为：" + str((np.sum(epipolar_disps_all))/len(epipolar_disps_all)))
#


if __name__ == '__main__':
    # for i in range(1, 12):
    #     print('\n\n***************' + str(i) + "***************")
    # # #     # # # GF7_our
    #     epiL = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/3/GF7_DLC_W88.1_N42.0_20210917_L1A0000565411_{}/cut/ImageL".format(str(i))
    #     epiR = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/3/GF7_DLC_W88.1_N42.0_20210917_L1A0000565411_{}/cut/ImageR".format(str(i))
    #     disp = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/3/GF7_DLC_W88.1_N42.0_20210917_L1A0000565411_{}/cut/Disp".format(str(i))
    #     save = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/3/GF7_DLC_W88.1_N42.0_20210917_L1A0000565411_{}/cut/check_F-5-1-2.txt".format(str(i))
    #
        # epiL = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/4/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412_{}/cut/ImageL".format(str(i))
        # epiR = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/4/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412_{}/cut/ImageR".format(str(i))
        # disp = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/4/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412_{}/cut/Disp".format(str(i))
        # save = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/4/GF7_DLC_W88.2_N41.8_20210917_L1A0000565412_{}/cut/check_F-5-1.txt".format(str(i))
    #
        # epiL = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/2/GF7_DLC_W87.9_N41.8_20210720_L1A0000864854_{}/cut/ImageL".format(str(i))
        # epiR = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/2/GF7_DLC_W87.9_N41.8_20210720_L1A0000864854_{}/cut/ImageR".format(str(i))
        # disp = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/2/GF7_DLC_W87.9_N41.8_20210720_L1A0000864854_{}/cut/Disp".format(str(i))
        # save = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/2/GF7_DLC_W87.9_N41.8_20210720_L1A0000864854_{}/cut/check_F-5-1.txt".format(str(i))
    #
        # epiL = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/1/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_{}/cut/ImageL".format(str(i))
        # epiR = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/1/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_{}/cut/ImageR".format(str(i))
        # disp = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/1/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_{}/cut/Disp".format(str(i))
        # save = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/1/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_{}/cut/check_F-5-1.txt".format(str(i))
        # sift_check_epipolar(epiL, epiR, disp, 4, save)

    # epiL = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_1-0/ImageL"
    # epiR = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_1-0/ImageR"
    # disp = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_1-0/Disp"
    # save = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.9_N42.0_20210720_L1A0000500901_1-0/check.txt"
    # sift_check_epipolar(epiL, epiR, disp, 6, save)

    # # our
    # epiL = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/*/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/*/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/*/Disp"
    # save = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/trainvaltest_check_Foundation_5-1-08_maxe20-.txt"
    epiL = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/valtest/ImageL"
    epiR = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/valtest/ImageR"
    disp = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/valtest/Disp"
    save = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res_new/valtest.txt"
    # epiL = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res/*/*/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res/*/*/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res/*/*/Disp"
    # save = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/good_res/check_Foundation2test.txt"

    # # # our SanFran
    # epiL = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/6/GF7_DLC_W122.3_N37.8_20241015_L1A0001730050_1/cut/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/6/GF7_DLC_W122.3_N37.8_20241015_L1A0001730050_1/cut/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/6/GF7_DLC_W122.3_N37.8_20241015_L1A0001730050_1/cut/Disp"
    # save = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/6/GF7_DLC_W122.3_N37.8_20241015_L1A0001730050_1/cut/check_F-5-1-2.txt"

    # # # # our Cook
    # epiL = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/5/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/5/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/5/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/Disp"
    # save = r"/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/11/5/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/check_F-5-1.txt"

    # # # # # # GF7 WHU_Stereo
    # epiL = r"/home/dshare/01Data/3DDisp/GF7/data/valtest/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/GF7/data/valtest/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/GF7/data/valtest/disp"
    # save = r"/home/dshare/01Data/3DDisp/GF7/data/valtest/check_F-5-2.txt"

    # # # # # Vanh - 3D
    # epiL = r"/home/dshare/01Data/3DDisp/VG/trainvaltest/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/VG/trainvaltest/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/VG/trainvaltest/Disp"
    # save = r"/home/dshare/01Data/3DDisp/VG/trainvaltest/check_F-5-1.txt"
    #
    # # # # US3D
    # epiL = r"/home/dshare/01Data/3DDisp/WV3D/data/valtest/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/WV3D/data/valtest/ImageR"
    # disp = r"/home/dshare/01Data/3DDisp/WV3D/data/valtest/disp"
    # save = r"/home/dshare/01Data/3DDisp/WV3D/data/valtest/check_F-5-2.txt"

    # # # GF7 BM
    # epiL = r"/home/dshare/01Data/3DDisp/GF7/BigModel/ImageL"
    # epiR = r"/home/dshare/01Data/3DDisp/GF7/BigModel/ImageR"
    # disp = None
    # save = r"/home/dshare/01Data/3DDisp/GF7/BigModel/check_F-5-15.txt"
    # # # GF7 BM
    # epiL = r"/home/dshare/01Data/3DDisp/GF7/BigModel/pair_1-GF1GF2"
    # epiR = r"/home/dshare/01Data/3DDisp/GF7/BigModel/pair_2-GF1GF2"

    sift_check_epipolar(epiL, epiR, disp, 4, save)
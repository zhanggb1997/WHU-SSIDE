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
import cv2
import numpy as np

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


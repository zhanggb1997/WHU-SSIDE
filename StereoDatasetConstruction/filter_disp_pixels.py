'''
Project    : RSDeploy
FileName   : filter_disp_pixels .py
CreateTime : 2025/7/20 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import glob
import os.path

import numpy as np
import cv2
from tqdm import tqdm


def fast_filter_image(image, n=7, threshold_count=20, scale=10):
    """
    高效滤波处理（向量化实现）
    :param image: 输入图像（灰度图）
    :param n: 窗口大小（奇数）
    :return: 滤波后的图像
    """
    if n % 2 == 0:
        raise ValueError("窗口大小n必须是奇数")

    r = n // 2
    # threshold_count = int(n * n * 0.9)  # 90%的像素数量

    # 1. 边界填充（高效处理边界）
    padded = cv2.copyMakeBorder(image, r, r, r, r, cv2.BORDER_REFLECT)

    # 2. 创建滑动窗口视图（无数据复制）
    windows = np.lib.stride_tricks.sliding_window_view(padded, (n, n))

    # 3. 提取中心像素值
    center_vals = image.astype(np.float32)

    # 4. 向量化比较：每个窗口内像素 > (中心像素 + 10)
    if scale > 0:
        comparison = ((windows > (center_vals[:, :, np.newaxis, np.newaxis] + scale)) & (windows!=-999)).astype(np.uint8)
    else:
        comparison = ((windows < (center_vals[:, :, np.newaxis, np.newaxis] + scale)) & (windows!=-999)).astype(np.uint8)

    # 5. 统计符合条件的像素数量
    count_map = np.sum(comparison, axis=(2, 3))

    # 6. 条件掩码：需要剔除的位置
    mask = (count_map >= threshold_count)

    # 7. 应用掩码（保留原图，仅剔除满足条件的点）
    result = image.copy()
    result[mask] = -999

    return result


# 使用示例
if __name__ == "__main__":

    disp_paths = glob.glob("/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/Disp/*")
    save_path  = "/home/dshare/01Data/3DDisp/LiDAR/America/epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut/Disp1"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for disp_path in tqdm(disp_paths):

        # 读取图像
        disp = cv2.imread(disp_path, -1)

        # 设置窗口大小
        # window_size = 7  # 可尝试不同大小的窗口

        # 优化版本
        result_optimized = fast_filter_image(disp, 7, 10, -10)

        # 保存结果
        cv2.imwrite(os.path.join(save_path, os.path.basename(disp_path)), result_optimized)


if __name__ == '__main__':
    pass

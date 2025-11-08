'''
Project    : RSDeploy
FileName   : downsample .py
CreateTime : 2025/7/16 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import glob
import numpy as np
import cv2

imgs_dir = r"/home/dshare/03Output/3DDisp/DenseMatch/Our/best-1_HMSM-Ori_HMSM_Our-A_EP1-40_BS1_LR0.001"
dips_dir = r"/home/dshare/03Output/3DDisp/DenseMatch/Our/best-1_HMSM-Ori_HMSM_Our-A_EP1-40_BS1_LR0.001"

ori_disps_path = glob.glob(imgs_dir)
ori_trues_path = glob.glob(dips_dir)

for disp_path, true_path in zip(ori_disps_path, ori_trues_path):
    disp = cv2.imread(disp_path, -1)
    true = cv2.imread(true_path, -1)

    resample_disp = cv2.medianBlur(disp, 5)
    resample_true = cv2.medianBlur(true, 5)



if __name__ == '__main__':
    pass

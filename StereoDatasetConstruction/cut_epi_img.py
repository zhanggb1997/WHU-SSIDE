'''
Project    : RSDeploy
FileName   : cut_epi_img .py
CreateTime : 2025/2/24 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import json
import math
import os
from copy import deepcopy
from datetime import datetime
import cv2
import xml.dom.minidom
import numpy as np

shift_ = 0

class XMLC():

    def __init__(self, xml_path):
        xmls = xml.dom.minidom.parse(xml_path)


        resolutions = {
            'GF01': 2.,
            'GF02': 1.,
            'GF06': 2.,
            'GF7BWD': 0.65,
            'GF7FWD': 0.80,
        }
        # 影像元信息
        if 0:
        # if xmls.getElementsByTagName('SceneID'):
            scene_info = xmls.getElementsByTagName('SceneID')[0].firstChild.data
            self.satellite = scene_info.split('_')[0]
            self.date = scene_info.split('_')[3]
            self.time = datetime.strptime(self.date, "%Y%m%d")
            self.gsp = resolutions[self.satellite]
        else:
            # 卫星名称
            if xmls.getElementsByTagName('SatelliteID'):
                self.satellite = xmls.getElementsByTagName('SatelliteID')[0].firstChild.data
            # 传感器名称
            if xmls.getElementsByTagName('SensorID'):
                self.sensor = xmls.getElementsByTagName('SensorID')[0].firstChild.data
            # 获取时间
            # if xmls.getElementsByTagName('CenterTime'):
            #     self.date = xmls.getElementsByTagName('CenterTime')[0].firstChild.data
            #     # 影像的获取时间规则化
            #     self.time = datetime.strptime(self.date, "%Y%m%d%H%M%S")
            if xmls.getElementsByTagName('StartTime'):
                self.date = xmls.getElementsByTagName('StartTime')[0].firstChild.data
                # 影像的获取时间规则化
                self.time = datetime.strptime(self.date, "%Y-%m-%d %H:%M:%S")
            # 分辨率
            self.gsp = resolutions[self.satellite + self.sensor]

        # 四个顶点的维度经度
        if xmls.getElementsByTagName('SceneCenterLong'):
            self.lat = float(xmls.getElementsByTagName('SceneCenterLat')[0].firstChild.data)
            self.lon = float(xmls.getElementsByTagName('SceneCenterLong')[0].firstChild.data)
        elif xmls.getElementsByTagName('UpperLeftLat'):
            self.lat = float(xmls.getElementsByTagName('UpperLeftLat')[0].firstChild.data)
            self.lon = float(xmls.getElementsByTagName('UpperLeftLong')[0].firstChild.data)
        elif xmls.getElementsByTagName('CenterLatitude'):
            self.lat = float(xmls.getElementsByTagName('CenterLatitude')[0].firstChild.data)
            self.lon = float(xmls.getElementsByTagName('CenterLongitude')[0].firstChild.data)
        else:
            print("经纬度未能成功识别！")

        # 分辨率

        # 太阳天顶角
        if xmls.getElementsByTagName('SolarElevation'):
            self.sunzenith = 90 - float(xmls.getElementsByTagName('SolarElevation')[0].firstChild.data)
        elif xmls.getElementsByTagName('SolarZenith'):
            self.sunzenith = float(xmls.getElementsByTagName('SolarZenith')[0].firstChild.data)

        # 太阳方位角
        if xmls.getElementsByTagName('SolarAzimuth'):
            self.sunazimuth = float(xmls.getElementsByTagName('SolarAzimuth')[0].firstChild.data)

        # 卫星天顶角
        if xmls.getElementsByTagName('SatelliteElevation'):
            self.zenith = 90 - float(xmls.getElementsByTagName('SatelliteElevation')[0].firstChild.data)
        elif xmls.getElementsByTagName('SatelliteZenith'):
            self.zenith = float(xmls.getElementsByTagName('SatelliteZenith')[0].firstChild.data)

        # 卫星方位角
        if xmls.getElementsByTagName('SatelliteAzimuth'):
            self.azimuth = float(xmls.getElementsByTagName('SatelliteAzimuth')[0].firstChild.data)



# def getImgPair(imgLPath, imgRPath, dispPath):
def getImgPair(imgLPath, imgRPath, dispPath=None, scale=1, xmlLPath=None, xmlRPath=None, maskPath=None):
    imgL = cv2.imread(imgLPath, flags=cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgRPath, flags=cv2.IMREAD_GRAYSCALE)
    disp = cv2.imread(dispPath, flags=cv2.IMREAD_UNCHANGED)

    if maskPath:
        mask = np.array(cv2.imread(maskPath, -1))
        mask_ = mask==1
        disp[mask_] = -999

    if scale == 1:
        pass
    else:
        imgL = cv2.resize(imgL, (round(imgL.shape[1] / scale), round(imgL.shape[0] / scale)))
        imgR = cv2.resize(imgR, (round(imgR.shape[1] / scale), round(imgR.shape[0] / scale)))

    if xmlLPath and xmlRPath:
        xmlL = XMLC(xmlLPath)
        xmlR = XMLC(xmlRPath)

        return imgL, imgR, disp, xmlL, xmlR

    return imgL, imgR, disp


def cut_epi_imgs(ImgL, ImgR, Disp, sub_size, repeat_size, xmlL=None, xmlR=None, save_path=None, save_suffix="ImgL_Dupage_GF7_F_B_1_1"):
    # 影像处理
    heightL, widthL = ImgL.shape[:2]
    heightR, widthR = ImgR.shape[:2]
    heightD, widthD = Disp.shape[:2]

    # assert widthL == widthR == widthD and heightL == heightR == heightD

    x_all = math.ceil(widthL / (sub_size - repeat_size))
    y_all = math.ceil((heightL - shift_) / (sub_size - repeat_size))

    base_pathL = os.path.join(save_path, "ImageL")
    base_pathR = os.path.join(save_path, "ImageR")
    base_pathD = os.path.join(save_path, "Disp")
    base_pathM = os.path.join(save_path, "Meta")

    if not os.path.exists(base_pathL):
        os.makedirs(base_pathL)
    if not os.path.exists(base_pathR):
        os.makedirs(base_pathR)
    if not os.path.exists(base_pathD):
        os.makedirs(base_pathD)
    if not os.path.exists(base_pathM):
        os.makedirs(base_pathM)


    # 元信息处理
    if xmlL and xmlR:
        # # 计算夹角（单位为弧度）
        # angle_radians = np.arccos(dot_product / (norm_left * norm_right))
        angle_radians = math.acos(math.sin(math.radians(xmlL.zenith)) * math.sin(math.radians(xmlR.zenith)) * math.cos(
            math.radians(xmlL.azimuth - xmlR.azimuth)) + math.cos(math.radians(xmlL.zenith)) * math.cos(math.radians(xmlR.zenith)))
        # 转换为角度
        angle_degrees = math.degrees(angle_radians)
        # angle_degrees = 15.

        # 时间差异处理
        time_diff = abs(xmlL.time - xmlR.time)
        # 计算月/日/时/分/秒的时间间隔差异
        days_diff = time_diff.days + time_diff.seconds / 3600 / 24
        seconds_diff = time_diff.days * 36000 * 24 + time_diff.seconds
        months_diff = days_diff / 30
        hours_diff = days_diff * 24
        minutes_diff = days_diff * 24 * 60

        # 写入的数据
        data_dict = {
            "left_date": str(xmlL.time),
            "right_date": str(xmlR.time),
            "month_diff": float(months_diff),
            "dsy_diff": float(days_diff),
            "hour_diff": float(hours_diff),
            "minute_diff": float(minutes_diff),
            "second_diff": float(seconds_diff),
            "left_lat": float(xmlL.lat),
            "left_lon": float(xmlL.lon),
            "right_lat": float(xmlR.lat),
            "right_lon": float(xmlR.lon),
            "left_cam": xmlL.satellite + xmlL.sensor,
            "right_cam": xmlR.satellite + xmlR.sensor,
            "left_gsp": float(xmlL.gsp),
            "right_gsp": float(xmlR.gsp),
            "left_zenith": float(xmlL.zenith),
            "right_zenith": float(xmlR.zenith),
            "left_azimuth": float(xmlL.azimuth),
            "right_azimuth": float(xmlR.azimuth),
            "stereo_angle": float(angle_degrees),
            "left_sunazimuth": float(xmlL.sunazimuth),
            "right_sunazimuth": float(xmlR.sunazimuth),
            "left_sunzenith": float(xmlL.sunzenith),
            "right_sunzenith": float(xmlR.sunzenith),
        }

    for x_no in range(x_all):
        w_start = max(0, x_no * (sub_size - repeat_size))
        if (x_no * (sub_size - repeat_size) + sub_size) >= widthL:
            w_start = max(0, widthL - sub_size)
        for y_no in range(y_all):
            h_start = max(0, y_no * (sub_size - repeat_size))
            if (y_no * (sub_size - repeat_size) + sub_size + shift_) >= heightL:
                h_start = max(0, heightL - sub_size - shift_)
            imgl_ = ImgL[h_start: h_start + sub_size, w_start: w_start + sub_size]
            disp_ = deepcopy(Disp[h_start: h_start + sub_size, w_start: w_start + sub_size])

            if np.max(disp_) == -999:
                continue
            else:
                eval_disp = np.mean(disp_, where=disp_ > -999)
                # w_start_r = min(max(0, round(w_start - eval_disp)), widthR - 1024)
                w_start_r = round(w_start - eval_disp)
                if w_start_r < 0:
                    imgr_ = np.zeros_like(imgl_)
                    imgr_[:, -w_start_r:] = ImgR[h_start + shift_: h_start + shift_ + sub_size, 0: w_start_r + sub_size]
                elif w_start_r > (widthR - sub_size):
                    shift_w = w_start_r - (widthR - sub_size)
                    imgr_ = np.zeros_like(imgl_)
                    imgr_[:, :-shift_w] = ImgR[h_start + shift_: h_start + shift_ + sub_size, -(sub_size - shift_w):]
                else:
                    imgr_ = ImgR[h_start + shift_: h_start + shift_ + sub_size, w_start_r: w_start_r + sub_size]
            disp_[disp_ > -999] = disp_[disp_ > -999] + (w_start_r - w_start)

            save_pathL = os.path.join(base_pathL, 'ImgL' + save_suffix + "_{}_{}.tif".format(h_start, w_start))
            save_pathR = os.path.join(base_pathR, 'ImgR' + save_suffix + "_{}_{}.tif".format(h_start, w_start))
            save_pathD = os.path.join(base_pathD, 'Disp' + save_suffix + "_{}_{}.tif".format(h_start, w_start))
            save_pathM = os.path.join(base_pathM, 'Meta' + save_suffix + "_{}_{}.json".format(h_start, w_start))

            cv2.imwrite(save_pathL, imgl_)
            cv2.imwrite(save_pathR, imgr_)
            cv2.imwrite(save_pathD, disp_)

            with open(save_pathM, "w", encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4, ensure_ascii=False)





def main_cut_epi_imgs(ImgLPath, ImgRPath, DispPath=None, sub_size=1024, repeat_size=128, SavePath=None, SaveSuffix="ImgL_Dupage_GF7_B_F_2", mask_path=None):
    # 读取影像
    ImgL, ImgR, Disp, xmlL, xmlR = getImgPair(ImgLPath, ImgRPath, DispPath, 1, xmlL_path, xmlR_path, mask_path)

    # 裁剪影像为指定大小
    cut_epi_imgs(ImgL, ImgR, Disp, sub_size, repeat_size, xmlL, xmlR, SavePath, SaveSuffix)



xmlL_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-BWDPAN.xml"
xmlR_path = r"../images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072/shp/new/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072-FWDPAN.xml"
imgL_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/LRD/eimgL_transp.tif"
imgR_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/LRD/eimgR_transp_transf.tif"
disp_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/LRD/epiDisp_transp.tif"
mask_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/LRD/mask.tif"
save_path = r"../epi_res/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072_0/cut"
main_cut_epi_imgs(imgL_path, imgR_path, disp_path, 1024, 512, save_path, "GF7_Cook_0_0_B_F", mask_path)

'''
Project    : RSDetec
FileName   : DM_data .py
CreateTime : 2023/8/26 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import json
import os
from glob import glob
import cv2
import numpy as np
from natsort import natsorted
from DenseMatchingModel.data.DM_data.En_data import Data_Normal_DM, RandomLocalGlobalEnhance, Data_Enhance_DM
from DenseMatchingModel.data.base.base_data import DataBase
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


class DataDM(DataBase):
    def __init__(self, cfgs, train_mode=0, have_path=False):
        super(DataDM, self).__init__(cfgs, train_mode, have_path)
        self.datasets_normal_L = Data_Normal_DM(mean=cfgs.DATA.MEAN_STD_L[train_mode][0], std=cfgs.DATA.MEAN_STD_L[train_mode][1])
        self.datasets_normal_R = Data_Normal_DM(mean=cfgs.DATA.MEAN_STD_R[train_mode][0], std=cfgs.DATA.MEAN_STD_R[train_mode][1])

    def define_datasets(self):
        if self.cfgs.DATA.HAVE_TXT_LIST:
            for mode_no, mode in enumerate(self.cfgs.DATA.MODES):
                self.datasets_mode[mode] = self.cfgs.DATA.READ_MODE[mode_no]
                if self.train_mode == 0 or self.train_mode == 'train':
                    txt_path = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TRAIN_FILE, self.cfgs.DATA.TRAIN_FILE + ".txt")
                    self.datasets_list[mode] = natsorted(self.get_data_txt(txt_path, 5.0, 10, mode, mode_no))
                elif self.train_mode == 1 or self.train_mode == 'val':
                    txt_path = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.VAL_FILE, self.cfgs.DATA.VAL_FILE + ".txt")
                    self.datasets_list[mode] = natsorted(self.get_data_txt(txt_path, 3.0, 10, mode, mode_no))
                elif self.train_mode == 2 or self.train_mode == 'test':
                    txt_path = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TEST_FILE, self.cfgs.DATA.TEST_FILE + ".txt")
                    self.datasets_list[mode] = natsorted(self.get_data_txt(txt_path, 3.0, 10, mode, mode_no))
        else:
            for mode_no, mode in enumerate(self.cfgs.DATA.MODES):
                self.datasets_mode[mode] = self.cfgs.DATA.READ_MODE[mode_no]
                if self.train_mode == 0 or self.train_mode == 'train':
                    self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TRAIN_FILE, mode)
                    self.datasets_list[mode] = natsorted(glob(os.path.join(self.datasets_path[mode], '*' + self.cfgs.DATA.READ_FORMAT[mode_no])))
                elif self.train_mode == 1 or self.train_mode == 'val':
                    self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.VAL_FILE, mode)
                    self.datasets_list[mode] = natsorted(glob(os.path.join(self.datasets_path[mode], '*' + self.cfgs.DATA.READ_FORMAT[mode_no])))
                elif self.train_mode == 2 or self.train_mode == 'test':
                    self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TEST_FILE, mode)
                    self.datasets_list[mode] = natsorted(glob(os.path.join(self.datasets_path[mode], '*' + self.cfgs.DATA.READ_FORMAT[mode_no])))

    def read_data(self):
        for mode in self.cfgs.DATA.MODES:
            self.data[mode] = cv2.imdecode(np.fromfile(self.datasets_list[mode][self.index], dtype=np.uint8), self.datasets_mode[mode])

        if self.cfgs.TRAIN.CONSISL:
            data_C = self.consis_transf_data(self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]], self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTR_ORDER]], self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]])
            self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER] + "_C"], self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTR_ORDER] + "_C"], self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER] + "_C"] = data_C[:3]
            self.data["mode_C"], self.data["LorR_C"], self.data["tx_C"], self.data["ty_C"] = data_C[3:]


    def get_data_txt(self, txt_path, disp_confidence, match_num_confidence, mode, id):
        data_list = []
        f = open(txt_path)
        lines = f.readlines()
        f.close()
        for line in lines:
            fileL_name, fileR_name, disp_name, disp_diff, match_num, good_match_num, ratio = line.split(" ")
            file_name = [fileL_name, fileR_name, disp_name]
            if disp_diff == 'nan':
                continue
            elif (float(disp_diff) < disp_confidence) and (int(match_num) > match_num_confidence):  # 查找符合条件的数据
                data_list.append(os.path.join(os.path.split(txt_path)[0], mode, file_name[id]))
        return data_list

    def get_data_txt2(self, txt_path, epi_conf_MAE, epi_conf_RMSE, disp_conf_MAE, disp_conf_RMSE, disp_num_T, mode, id):
        data_list = []
        f = open(txt_path)
        lines = f.readlines()
        f.close()
        for line in lines:
            fileL_name, fileR_name, disp_name, epi_pot_num, epi_err_MAE, epi_err_RMSE, epi_err_sum, epi_err_max, epi_err_min, disp_pot_num, disp_err_MAE, disp_err_RMSE, disp_err_sum, disp_err_max, disp_err_min = line.split(" ")
            file_name = [fileL_name, fileR_name, disp_name]

            if (float(epi_err_MAE) < epi_conf_MAE) and (float(epi_err_RMSE) < epi_conf_RMSE) and (float(disp_err_MAE) < disp_conf_MAE) and (float(disp_err_RMSE) < disp_conf_RMSE) and (int(disp_pot_num) > disp_num_T):  # 查找符合条件的数据
                data_list.append(os.path.join(os.path.split(txt_path)[0], mode, file_name[id]))
        return data_list


    def check_data(self):
        # ==============检查通道数量并进行规范化==============
        for mode in self.data.keys():
            if mode in ['Path', 'sampled_cols', 'sampled_rows', 'sampled_cols_T', 'sampled_rows_T', 'disp_x16', 'disp_x8', 'disp_x4', 'meta', 'Meta', "mode_C", "LorR_C", "tx_C", "ty_C"]:
                pass
            else:
                if len(self.data[mode].shape) == 3:  # [H x W x C] ==> [C x H x W]
                    self.data[mode] = np.transpose(self.data[mode], (2, 0, 1))
                if len(self.data[mode].shape) == 2:  # [H x W] ==> [1 x H x W]
                    self.data[mode] = np.expand_dims(self.data[mode], 0)
                if not 'DI' in mode:
                    assert self.data[mode].shape[1] == self.cfgs.TRAIN.IMAGE_SIZE, mode + ':图像大小不一致！'

    def add_data(self):
        if self.have_path:
            self.data['Path'] = os.path.basename(self.datasets_list[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]][self.index])
        if self.cfgs.MODEL.NAME in ["HMSM", "OURNetV2", "OURNetV3", 'DMBigModelMeta', 'DMMeatMRGE']:
            if self.cfgs.MODEL.NUM_INCHANNELS != 1:
                L_gray = self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]][0]
            else:
                L_gray = self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]]
            self.data['dx'] = sig.convolve2d(np.squeeze(L_gray), kx, 'same')
            self.data['dy'] = sig.convolve2d(np.squeeze(L_gray), ky, 'same')


        if self.cfgs.DATA.ADD_META:
            self.data[self.cfgs.DATA.META_FILE] = dict()
            if 'WV' in self.cfgs.DATA.NAME or 'GF' in self.cfgs.DATA.NAME:
                meta_path = (self.datasets_list[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]][self.index]).replace(self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER], self.cfgs.DATA.META_FILE).replace(self.cfgs.DATA.READ_FORMAT[self.cfgs.DATA.INPUTL_ORDER], self.cfgs.DATA.META_FORMAT)
            if 'Our' in self.cfgs.DATA.NAME:
                meta_path = (self.datasets_list[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]][self.index]).replace(self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER], self.cfgs.DATA.META_FILE).replace('tif', self.cfgs.DATA.META_FORMAT)
            if 'GF' in self.cfgs.DATA.NAME:
                meta_path = meta_path.replace('left', 'meta')
            if 'Our' in self.cfgs.DATA.NAME:
                meta_path = meta_path.replace('ImgL', 'Meta')
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                for meta_mode_no, meta_mode in enumerate(self.cfgs.DATA.META_MODE):
                    self.data[self.cfgs.DATA.META_FILE][meta_mode] = ((meta_data[meta_mode]) / self.cfgs.DATA.META_MODE_SCALE[meta_mode_no]) + self.cfgs.DATA.META_MODE_SHIFT[meta_mode_no]


if __name__ == '__main__':
    pass

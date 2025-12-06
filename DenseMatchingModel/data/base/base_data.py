'''
Project    : RSDetec
FileName   : data_train .py
CreateTime : 2023/8/24 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import os
from glob import glob
import cv2
import numpy as np
import torch
from colorama import Fore
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import albumentations as A
from configs.Segment.default import defrost_config, freeze_config


# 数据集
class DataBase(Dataset):
    def __init__(self, cfgs, train_mode=0, have_path=False):
        self.cfgs = cfgs
        self.train_mode = train_mode
        self.datasets_mode = dict()
        self.datasets_path = dict()
        self.datasets_list = dict()
        self.have_path = have_path
        self.define_datasets()
        self.datasets_tran_enhan()

    def define_datasets(self):
        for mode_no, mode in enumerate(self.cfgs.DATA.MODES):
            self.datasets_mode[mode] = self.cfgs.DATA.READ_MODE[mode_no]
            if self.train_mode == 0 or self.train_mode == 'train':
                self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TRAIN_FILE, mode)
            elif self.train_mode == 1 or self.train_mode == 'val':
                self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.VAL_FILE, mode)
            elif self.train_mode == 2 or self.train_mode == 'test':
                self.datasets_path[mode] = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TEST_FILE, mode)

            self.datasets_list[mode] = natsorted(glob(os.path.join(self.datasets_path[mode], '*.' + self.cfgs.DATA.READ_FORMAT[mode_no])))

    def __len__(self):
        return len(self.datasets_list[self.cfgs.DATA.MODES[0]])

    def __getitem__(self, index):
        self.data = dict()
        self.index = index
        self.read_data()
        self.transf_data()
        self.add_data()
        self.check_data()

        return self.data


    def transf_data(self):
        if self.cfgs.MODE == "Train" or self.cfgs.MODE == "train":
            En_data = self.datasets_Aenhangce(image=self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]],
                                              mask=self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]])
            self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]] = En_data["image"]
            self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]] = En_data["mask"]
        elif self.cfgs.MODE == "Pred" or self.cfgs.MODE == "pred":
            En_data = self.datasets_Aenhangce(image=self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]])
            self.data[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]] = En_data["image"]
        for mode_no, mode in enumerate(self.cfgs.DATA.MODES):
            self.data[mode] = (self.data[mode] / self.cfgs.DATA.SCALE[mode_no] + self.cfgs.DATA.SHIFT[mode_no]).astype(np.float32)


    def datasets_tran_enhan(self):
        self.datasets_Aenhangce = A.Compose([
            A.CLAHE(clip_limit=3, p=1.0),
                                            ])



class LoadBase(DataLoader):
    def __init__(self, cfgs, Dataset):
        self.cfgs = cfgs
        self.load_mode = Dataset.train_mode
        self.dataset = Dataset
        self.sub_res = np.zeros(cfgs.TRAIN.BATCH_SIZE, dtype=np.uint8)
        super(LoadBase, self).__init__(Dataset, batch_size=self.cfgs.TRAIN.BATCH_SIZE, pin_memory=True,
                                       shuffle=self.cfgs.DATA.SHUFFLE, drop_last=self.cfgs.DATA.DROPLAST, num_workers=0)

    def toTqdm(self, color):
        return tqdm_iterator(self, color)

def tqdm_iterator(iterator, color="red"):
    color_mode = None
    try:
        if color == "red":
            color_mode = Fore.RED
        elif color == "green":
            color_mode = Fore.GREEN
        elif color == "blue":
            color_mode = Fore.BLUE
        else:
            raise ColorError(color)
    except ColorError as e:
        print("<{}> <blue>, <red>, <green>".format(e.value))
    tqdm_iter = tqdm(iterator, ncols=250)
    tqdm_iter.bar_format = '{l_bar}%s{bar}%s{r_bar}' % (color_mode, Fore.RESET)
    tqdm_iter.unit = 'iterate'
    setattr(tqdm_iter, 'len', len(iterator.dataset))
    setattr(tqdm_iter, 'load_mode', iterator.load_mode)
    setattr(tqdm_iter, 'batch_size', iterator.batch_size)
    return tqdm_iter


class ColorError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



if __name__ == '__main__':
    pass

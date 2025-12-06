'''
Project    : RSDetec
FileName   : base_test .py
CreateTime : 2023/8/25 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import os
import time
import cv2
import numpy as np
import torch
from core.base.base_train import TrainBase
from data.base.base_data import DataBase, LoadBase
from utils.log import logS
from eval.base.eval_utils import calculate_all

class TestBase(TrainBase):
    def __init__(self, cfgs, model):
        super(TestBase, self).__init__(cfgs, model)

    def data_init(self):
        self.data_in = dict()  # 输入数据
        self.data_out = []  # 输出数据

        self.data_t = DataBase(self.cfgs, 2, have_path=True)  # 测试数据
        self.load_t = LoadBase(self.cfgs, self.data_t)  # 测试数据加载器

        # 每轮的tqdm数据加载器
        self.tqdm_loader = None  # 数据加载器

    def test_model(self):
        torch.cuda.empty_cache()
        self.model.train(False)
        self.model.eval()
        logS.info('开始对测试集进行测试！')  # 输出开始测试
        self.timing.start()  # 开始计时
        with torch.no_grad():
            load_t = self.load_t
            self.train_model_epoch(load_t)  # test
        print('Training complete in {}'.format(str(self.timing)))
        logS.info('开始进行精度评估！')
        self.calculate_precise()

    def train_model_epoch(self, loader):
        self.tqdm_loader = loader.toTqdm('red')
        # 初始化一些属性参数
        step = 0
        # 馈送数据
        for data_dict in self.tqdm_loader:
            step += 1  # 当前批次数
            self.tqdm_loader.set_description('epoch:{}-test:{}'.format(self.epoch + 1, step))  # 设置tqdm左边显示内容
            # 数据拷贝到device中
            self.feed_network(data_dict)
            # 更新信息
            self.update_info()
            # 输出
            self.pred_out(self.data_out, data_dict['Path'])

    def pred_out(self, pred_lists, base_names):

        # 输出路径处理
        save_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG)  # 保存路径
        if not os.path.isdir(save_path):  # 如果不存在文件夹则创建
            os.makedirs(save_path)
        # 模型预测输出
        if self.cfgs.TRAIN.MUL_LOSS:  # 判断是不是多损失输出
            pred_lists = np.squeeze(pred_lists[-1].cpu().detach().numpy())
        pred_lists_cpu = np.squeeze(pred_lists.cpu().detach().numpy())  # 输出结果转换为cpu的np数值

        # 依据bs大小依次处理输出
        for ith, base_name in enumerate(base_names):
            out = np.zeros((pred_lists.shape[-2], pred_lists.shape[-1], 1 if self.cfgs.MODEL.NUM_CLASSES < 3 else 3), dtype=np.uint8)
            out = np.squeeze(out)
            # 二分类
            if self.cfgs.MODEL.NUM_CLASSES < 3:
                if len(base_names) == 1:
                    out[pred_lists_cpu[:, :] > 0.6] = 255
                else:
                    out[pred_lists_cpu[ith, :, :] > 0.6] = 255
            else:
                # 多分类
                assert len(TestBase.COLOR_DICT) == self.cfgs.MODEL.NUM_CLASSES
                if len(pred_lists_cpu.shape) == 3:
                    pred_lists_cpu = pred_lists_cpu[None, :, :, :]
                pred_list_cpu = pred_lists_cpu[ith]
                out = (np.argmax(pred_list_cpu, axis=0)).astype(np.uint8)
                # out = TestBase.COLOR_DICT[np.argmax(pred_list_cpu, axis=0)].astype(np.uint8)

            # # 闭运算
            out = cv2.dilate(out, np.ones((5, 5), np.uint8))
            out = cv2.erode(out, np.ones((5, 5), np.uint8))
            # # 开运算
            out = cv2.erode(out, np.ones((2, 2), np.uint8))
            out = cv2.dilate(out, np.ones((2, 2), np.uint8))

            self.pred_save(save_path, base_name, out)

    def pred_save(self, path, file_name, result):
        file_name = os.path.splitext(file_name)[0] + os.path.splitext(file_name)[1]
        cv2.imencode(os.path.splitext(file_name)[1], result)[1].tofile(os.path.join(path, file_name))

    def calculate_precise(self):
        pred_dir = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG)
        # true_dir = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.cfgs.DATA.GROUND_TRUTH)
        true_dir = os.path.join(self.cfgs.DATA.DATA_DIR, self.cfgs.DATA.TEST_FILE, self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER])
        now_time = time.strftime('%Y%m%d%H%M')
        txt_dir = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG + '_' + now_time + '.txt')

        if self.cfgs.MODEL.NUM_CLASSES < 3:
            calculate_all(true_dir, pred_dir, txt_dir, self.FLAG + '_预测结果', is_gray=True)
        else:
            calculate_all(true_dir, pred_dir, txt_dir, self.FLAG + '_预测结果', TestBase.COLOR_DICT)

    def feed_network(self, data_dict):
        for mode in data_dict.keys():
            if mode == 'Path':
                pass
            else:
                self.data_in[mode] = data_dict[mode].to(device=torch.device('cuda' if self.cfgs.GPUS else 'cpu'), dtype=torch.float32,
                                                      non_blocking=True)
        self.data_out = self.model(self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]])  # 预测

        if self.data_out.shape[-1] != self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUT_ORDER]].shape[-1]:
            self.data_out = torch.nn.functional.interpolate(self.data_out, size=self.data_in[self.cfgs.DATA.MODES[
                self.cfgs.DATA.TRUTH_ORDER]].shape[2:], mode='nearest', align_corners=None)


if __name__ == '__main__':
    pass

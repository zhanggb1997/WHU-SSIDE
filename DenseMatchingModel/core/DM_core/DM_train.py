'''
Project    : RSDetec
FileName   : DM_train .py
CreateTime : 2023/8/26 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import os

import numpy as np
import torch
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.nn import functional as F

from DenseMatchingModel.core.DM_core.DM_losses import compute_rr_loss, ContrastLossW, LossCalculator, XiShuSmoothL1Loss, \
    DispDistributionBCELoss, ConsisLoss
from DenseMatchingModel.core.base.base_train import TrainBase
from DenseMatchingModel.data.DM_data.DM_data import DataDM
from DenseMatchingModel.data.base.base_data import LoadBase
from DenseMatchingModel.utils.log import logS
from DenseMatchingModel.utils.tools import Acc_Loss_Dict, Acc_Loss_Record


class TrainDM(TrainBase):
    def __init__(self, cfgs, model):
        super(TrainDM, self).__init__(cfgs, model)

    def data_init(self):
        self.data_in = dict()  # 输入数据
        self.data_out = []  # 输出数据
        # 指定的数据加载器
        if self.cfgs.MODE in ["Train", "train", "TRAIN"]:
            self.data_t = DataDM(self.cfgs, 0)  # 训练数据
            self.load_t = LoadBase(self.cfgs, self.data_t)  # 训练数据加载器
            self.data_v = DataDM(self.cfgs, 1)  # 验证数据
            self.load_v = LoadBase(self.cfgs, self.data_v)  # 验证数据加载器
        elif self.cfgs.MODE in ["Test", "test", "TEST", "Pred", "pred", "PRED"]:
            self.data_t = DataDM(self.cfgs, 2, have_path=True)  # 测试数据
            self.load_t = LoadBase(self.cfgs, self.data_t)  # 测试数据加载器
        else:
            logS.error("config配置中训练Mode错误！")
        # 每轮的tqdm数据加载器
        self.tqdm_loader = None  # 数据加载器

    def criterions_init(self):
        self.criterions = {}
        for argum in self.cfgs.TRAIN.ARGUMENTS:
            if 'Acc' in argum:
                self.criterions[argum] = nn.MSELoss(reduction='mean')  # 精度
            elif 'Loss' in argum:
                if self.cfgs.MODEL.NAME == "OURNetV2":
                    self.criterions[argum] = nn.SmoothL1Loss()
                    self.criterions[argum + "_dm"] = nn.SmoothL1Loss()
                    self.criterions[argum + "_contrast"] = ContrastLossW(self.cfgs.TRAIN.MIN_DISP, self.cfgs.TRAIN.MAX_DISP)
                elif self.cfgs.MODEL.NAME == "HD3Net":
                    self.criterions[argum] = LossCalculator("stereo")
                elif self.cfgs.MODEL.NAME == "IGEV":
                    self.criterions[argum] = nn.SmoothL1Loss(reduction='mean')
                elif self.cfgs.TRAIN.CONSISL:
                    self.criterions[argum] = nn.SmoothL1Loss()
                    # self.criterions[argum + "_C"] = nn.L1Loss()
                    self.criterions[argum + "_C"] = ConsisLoss()
                else:
                    self.criterions[argum] = nn.SmoothL1Loss()
                    # self.criterions[argum + "_XS"] = XiShuSmoothL1Loss()
                # GraftNet Adapt
                if "graft" in self.cfgs.FLAG or "GraftNet" in self.cfgs.FLAG:
                    self.criterions[argum + "_Graft"] = DispDistributionBCELoss()

            elif 'rr' in argum:
                self.criterions[argum] = compute_rr_loss()
            else:
                logS.error("TRAIN.ARGUMENTS 参数不正确！")

    def feed_network(self, data_dict):
        for mode in data_dict.keys():
            if isinstance(data_dict[mode], dict):
                for sub_mode in data_dict[mode].keys():
                    data_dict[mode][sub_mode] = data_dict[mode][sub_mode].to(device=torch.device('cuda' if self.cfgs.GPUS else 'cpu'), dtype=torch.float16, non_blocking=True)
                self.data_in[mode] = data_dict[mode]
            else:
                self.data_in[mode] = data_dict[mode].to(device=torch.device('cuda' if self.cfgs.GPUS else 'cpu'), dtype=torch.float32, non_blocking=True)
        if self.cfgs.MODEL.NAME == "DMMeatMRGE":
            inputs = {'x_l': self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]], 'x_r': self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTR_ORDER]], 'dx': self.data_in['dx'], 'dy': self.data_in['dy'], 'meta': self.data_in[self.cfgs.DATA.META_FILE] if self.cfgs.DATA.ADD_META else None}
        else:
            inputs = {'x_l': self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTL_ORDER]], 'x_r':self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.INPUTR_ORDER]]}
        if self.cfgs.TRAIN.IS_AMP:  # 如果开启amp混合精度
            with autocast():
                self.data_out = self.model(inputs)  # 预测
        else:
            self.data_out = self.model(inputs)  # 预测


    def calculate_eval(self):
        self.step_acc_loss.values2zeros()
        if self.cfgs.TRAIN.MUL_LOSS:  # 判断是不是多损失输出
            out_num = len(self.cfgs.MODEL.DOWN_SCALE)  # 多损失的话，多损失的数量
        else:
            out_num = 1

        # DM Loss
        for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
            truth = self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]]  # 真值
            if self.cfgs.MODEL.NAME in ["DMMeatMRGE"]:
                if '-1' in mode and self.cfgs.TRAIN.MUL_LOSS:
                    mask = (self.cfgs.TRAIN.MIN_DISP <= truth) & (self.cfgs.TRAIN.MAX_DISP > truth)
                    pred = self.data_out[1][-1]
                    self.step_acc_loss.dicts[mode] += self.criterions[mode](pred[mask], truth[mask])
                else:
                    aggr_preds = self.data_out[0]
                    iter_preds = self.data_out[1]

                    min_disps = [self.model.s_min_disp, self.model.m_min_disp, self.model.l_min_disp]
                    max_disps = [self.model.s_max_disp, self.model.m_max_disp, self.model.l_max_disp]

                    for pred_no in range(out_num):  # 每个预测输出进行评估，计算损失
                        aggr_pred = aggr_preds[pred_no]  # 预测结果
                        truth_ = truth
                        mask = (min_disps[pred_no] <= truth) & (max_disps[pred_no] > truth)  # 参与运算的掩膜
                        self.step_acc_loss.dicts[mode] += self.criterions[mode](aggr_pred[mask], truth_[mask]) * (self.cfgs.TRAIN.MUL_LOSS_ALPHA[pred_no-1] if self.cfgs.TRAIN.MUL_LOSS else 1.0)

                    loss_gamma = 0.9
                    num_iter_pred = len(iter_preds)
                    mask = (self.cfgs.TRAIN.MIN_DISP <= truth) & (self.cfgs.TRAIN.MAX_DISP > truth)  # 参与运算的掩膜
                    for pred_no in range(num_iter_pred):  # 每个预测输出进行评估，计算损失
                        adjusted_loss_gamma = loss_gamma ** (15 / (num_iter_pred - 1))
                        i_weight = adjusted_loss_gamma ** (num_iter_pred - pred_no - 1)
                        # i_weight = 0
                        i_loss = (iter_preds[pred_no] - truth).abs()
                        assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, truth.shape, iter_preds[pred_no].shape]
                        self.step_acc_loss.dicts[mode] += i_weight * i_loss[mask.bool()].mean()
            self.epoch_acc_loss.dicts[mode] += float(self.step_acc_loss.dicts[mode]) * self.tqdm_loader.batch_size


    def optimizer_init(self):
        param_dicts = self.model.parameters()
        if self.cfgs.TRAIN.OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(param_dicts, lr=self.cfgs.TRAIN.LEARN_RATE, weight_decay=self.cfgs.TRAIN.WEIGHT_DECAY, betas=self.cfgs.TRAIN.BETAS)
        elif self.cfgs.TRAIN.OPTIMIZER == 'AdamW':
            self.optimizer = optim.AdamW(param_dicts, lr=self.cfgs.TRAIN.LEARN_RATE, weight_decay=self.cfgs.TRAIN.WEIGHT_DECAY, betas=self.cfgs.TRAIN.BETAS)
        elif self.cfgs.TRAIN.OPTIMIZER == 'RMSProp':
            self.optimizer = optim.RMSprop(param_dicts, lr=self.cfgs.TRAIN.LEARN_RATE, weight_decay=self.cfgs.TRAIN.WEIGHT_DECAY)
        elif self.cfgs.TRAIN.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(param_dicts, lr=self.cfgs.TRAIN.LEARN_RATE, weight_decay=self.cfgs.TRAIN.WEIGHT_DECAY)

        self.decay_step = 0
        self.decay_gama = 0
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000000, 1.0)
        if self.cfgs.TRAIN.DECAY_STEP:
            self.decay_step = self.cfgs.TRAIN.DECAY_STEP
            self.decay_gama = self.cfgs.TRAIN.DECAY_GAMMA
            self.warm_up_iter = self.cfgs.TRAIN.WARM_UPITER
            lambda_LR = lambda cur_iter: (cur_iter / (2 * self.warm_up_iter)) if cur_iter < self.warm_up_iter else (
                    self.decay_gama ** ((cur_iter - self.warm_up_iter) // self.decay_step))
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_LR, verbose=True)

    def ckpt_init(self):
        """
        """
        # pass
        # # ckpt_dir存在
        if self.cfgs.RESUME.CKPT_DIR:
            logS.info("Reading {} ...".format(self.cfgs.RESUME.CKPT_DIR))
            assert os.path.exists(self.cfgs.RESUME.CKPT_DIR), logS.error("配置文件中输入的ckpt文件不存在！")
            # 读取ckpt
            self.ckpt = torch.load(self.cfgs.RESUME.CKPT_DIR, map_location=torch.device('cpu'), weights_only=False)
            # ckpt中的信息读取
            # self.cfgs = self.ckpt['cfgs']
            model_weight = self.ckpt['model']
            # TorchDPP除去module
            if any(key.startswith("module.") for key in model_weight.keys()):
                for key, value in list(model_weight.items()):
                    # 如果键名以 "module." 开头，去掉前缀
                    if key.startswith("module."):
                        new_key = key.replace("module.", "")
                        model_weight[new_key] = value
                        del model_weight[key]
            self.model.load_state_dict(model_weight, False)
            self.record_acc_loss = self.ckpt["record_acc_loss"]
            if self.cfgs.RESUME.INIT_EPOCHES:
                self.epoch = 1
            else:
                self.epoch = self.ckpt['epoch'] + 1
            if not self.cfgs.RESUME.INIT_LR_DECAY:
                self.optimizer.load_state_dict(self.ckpt['optimizer'])
                self.lr_scheduler.load_state_dict(self.ckpt['lr_scheduler'])
                # pass
            else:
                self.decay_step = self.cfgs.TRAIN.DECAY_STEP
                self.decay_gama = self.cfgs.TRAIN.DECAY_GAMMA
                self.warm_up_iter = self.cfgs.TRAIN.WARM_UPITER
                lambda_LR = lambda cur_iter: (cur_iter / (2 * self.warm_up_iter)) if cur_iter < self.warm_up_iter else (self.decay_gama ** ((cur_iter - self.warm_up_iter) // self.decay_step))
                # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_LR)
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_LR, verbose=True)
                # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfgs.TRAIN.DECAY_STEP, gamma=self.cfgs.TRAIN.DECAY_GAMMA, verbose=True)
            # 是否重新设定best-loss
            if not self.cfgs.RESUME.INIT_BESTLOSS:
                for key in self.best_acc_loss.dicts.keys():
                    self.best_acc_loss.dicts[key] = np.min(list(self.record_acc_loss.dicts['val'][key].values()))

        else:
            self.ckpt = {
                "cfgs": self.cfgs,
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "record_acc_loss": self.record_acc_loss
            }

    def accloss_init(self):
        """
        精度损失初始化
        """
        self.best_acc_loss = Acc_Loss_Dict(self.cfgs, is_initial_loss_acc=True)  # 最佳精度-损失
        self.step_acc_loss = Acc_Loss_Dict(self.cfgs)  # 初始化每个step精度-损失值情况
        self.epoch_acc_loss = Acc_Loss_Dict(self.cfgs)  # 初始化轮次epoch的综合/平均损失值
        self.record_acc_loss = Acc_Loss_Record(self.cfgs)  # 整个训练过程的精度-损失进行记录

        if self.cfgs.MODEL.NAME == "OURNetV2":
            for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
                self.add_loss_key(mode, "_dm")
                self.add_loss_key(mode, "_contrast")

        if "graft" in self.cfgs.FLAG or "GraftNet" in self.cfgs.FLAG:
            for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
                if not "-1" in mode:
                    self.add_loss_key(mode, "_Graft")

        if self.cfgs.TRAIN.CONSISL:
            for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
                if not "-1" in mode:
                    self.add_loss_key(mode, "_C")
                    self.add_loss_key(mode, "_D")

        if self.cfgs.TRAIN.COST24X:
            for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
                if not "-1" in mode:
                    self.add_loss_key(mode, "_2X")
                    self.add_loss_key(mode, "_4X")




if __name__ == '__main__':
    pass

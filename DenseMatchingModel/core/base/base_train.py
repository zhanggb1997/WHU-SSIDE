'''
Project    : RSDetec
FileName   : BaseTrain .py
CreateTime : 2023/8/22 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现Train功能的基础类 #
'''
import os
import shutil
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from DenseMatchingModel.core.base.base_losses import MulClass_focal_loss, DSCLoss, TwoClass_focal_loss
from DenseMatchingModel.data.base.base_data import DataBase, LoadBase
from DenseMatchingModel.utils.log import logS
from DenseMatchingModel.utils.log.logs import show_table_log
from DenseMatchingModel.utils.tools import Acc_Loss_Dict, Acc_Loss_Record, Timing

class TrainBase():
    """
    The base class for train a model.
    """
    def __init__(self, cfgs, model):
        """
        初始化Train类，定义详细的配置参数
        :param cfgs: 配置参数
        :param model: 需要进行训练的模型
        """
        self.cfgs = cfgs  # 配置信息
        self.model = model  # 训练模型
        self.outdir_init()  # 配置输出文件夹初始化
        self.epoch_init()  # 训练轮次初始化
        self.model_init()  # 模型设置初始化
        self.data_init()  # 数据加载初始化
        self.accloss_init()  # 精度和损失初始化
        self.criterions_init()  # 评价函数初始化
        self.optimizer_init()  # 优化器初始化
        self.flag_init()  # 训练flag初始化
        self.ckpt_init()  # checkpoint初始化
        self.check_out()  # 检查输出文件夹是否有重名文件
        self.log_init()  # 输出Log初始化
        self.time_init()  # 计时初始化
        self.amp_init()  # 混合精度amp初始化
        self.mul_GPUs()  # 是否使用多GPU自动判断
        self.check_Info()  # 检查信息


    def check_Info(self):
        """
        输出相应的配置信息进行检查
        :return:
        """
        info_dict = self.cfg2dict(self.cfgs)
        show_table_log(" Task Info Print & Check ", l_len=30, r_len=50, info_dict=info_dict)

    def cfg2dict(self, cfg):
        info_dict = dict()
        for key, value in cfg.items():
            if value is not None:
                if key in info_dict.keys():
                    key = key + "_"
                if isinstance(value, dict):
                    info_dict[key] = ""
                    info_dict.update(self.cfg2dict(value))
                else:
                    info_dict[key] = value
        return info_dict



    def mul_GPUs(self):
        """
        是否多GPU并行训练
        :return:
        """
        # GPU数量大于1则进行多GPU运行
        if self.cfgs.GPUS > 1:
            device_ids = [i for i in range(self.cfgs.GPUS)]
            # 模型多GPU化
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)


    def amp_init(self):
        """
        混合精度amp初始化
        """
        # 如果开启amp
        if self.cfgs.TRAIN.IS_AMP:
            # # 创建GradScaler对象
            self.scaler = GradScaler()

    def log_init(self):
        """
        对需要存储的log进行初始化
        """
        self.log_write = SummaryWriter(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LOG, self.FLAG),
                                   comment=self.cfgs.TRAIN.COMMENT)  # 构建可视化损失和精确度图

    def check_out(self):
        """
        检查输出文件夹是否重名，存在
        """
        if self.cfgs.MODE in ["calcul", "Calcul"]:
            pass
        else:
            if os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG)):
                logS.warning("存在以往相同的训练FLAG输出结果，且并不是继续训练，请确认是否继续执行！y ：n")
                msg = input("请输入信息：")
                if msg in ['y', 'Y']:
                    if os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG)):
                        shutil.rmtree(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT, self.FLAG))
                else:
                    pass

    def time_init(self):
        """
        计时初始化
        """
        self.timing = Timing(False)

    def flag_init(self):
        """
        训练Flag设定
        """
        # Flag信息
        self.FLAG = self.cfgs.FLAG + '_' + self.cfgs.MODEL.NAME + '_' + self.cfgs.DATA.NAME + '_EP' + str(
            self.cfgs.TRAIN.START_EPOCH) + '-' + str(self.cfgs.TRAIN.STOP_EPOCH) + '_BS' + str(self.cfgs.TRAIN.BATCH_SIZE) + '_LR' + str(
            self.cfgs.TRAIN.LEARN_RATE)
        # 打印Flag
        logS.info(self.FLAG)


    def model_init(self):
        """
        模型初始化，并确定模型训练环境
        """
        if self.cfgs.GPUS == 0:
            self.model = self.model.to('cpu')
        elif self.cfgs.GPUS == 1:
            self.model = self.model.to('cuda')
        elif self.cfgs.GPUS > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,' + str(self.ckpt.GPUS - 1)
            self.model = self.model.to('cuda')
            self.model = nn.parallel.DistributedDataParallel(self.model)
        else:
            logS.error("输入GPUs数量错误！")


    def epoch_init(self):
        """
        初始化轮次信息
        """
        self.start_epoch = self.cfgs.TRAIN.START_EPOCH  # 开始训练的轮次
        self.stop_epoch = self.cfgs.TRAIN.STOP_EPOCH  # 结束训练的轮次
        self.epoch = self.start_epoch  # 当前轮次




    def outdir_init(self):
        """
        创建输出文件夹，用于确保输出文件夹存在
        """
        logS.info("正在初始化输出文件夹！")
        if not os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT)):
            os.mkdir(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LABEL_OUT))
        if not os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.ACCLOSS)):
            os.mkdir(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.ACCLOSS))
        if not os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LOG)):
            os.mkdir(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.LOG))
        if not os.path.exists(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT)):
            os.mkdir(os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT))
        logS.success("创建输出文件夹成功！")


    def train_model(self):
        # 清除缓存
        torch.cuda.empty_cache()
        # 开始训练
        logS.info('开始训练和验证')  # 打印标题
        self.timing.start()  # 计时器开始计时
        # 　开始训练验证
        for self.epoch in range(self.epoch, self.stop_epoch + 1):
            logS.info('Epoch {} / {}'.format(self.epoch, self.stop_epoch))
            for data_loader in (self.load_t, self.load_v):
            # for data_loader in (self.load_v,):
                self.train_model_epoch(data_loader)  # 训练
            self.lr_scheduler.step()  # 学习率动态调整
            self.save_ckpt(save_best=True, save_epoch=self.cfgs.TRAIN.SAVE_EPOCH,
                          save_per_epoch=self.cfgs.TRAIN.SAVE_PER_EPOCH)  # ckpt保存
        # 计时结束, 输出时间
        logS.success("训练成功完成, 训练时长：" + str(self.timing))
        # 记录绘制损失情况
        self.record_plot()

    # 训练 / 验证 一轮
    def train_model_epoch(self, data_loader):
        # 初始化一些属性参数
        self.step = 0  # 当前数据step
        time_t = Timing()  # 计时器
        # 数据加载器
        self.tqdm_loader = data_loader.toTqdm("blue" if data_loader.load_mode == 0 else 'green')
        # acc_loss值归零
        self.epoch_acc_loss.values2zeros()
        # 根据data loader执行训练模式
        self.model.train(self.tqdm_loader.load_mode == 0)
        # self.model.eval()
        if self.tqdm_loader.load_mode != 0 : self.model.eval()
        with torch.enable_grad() if self.tqdm_loader.load_mode == 0 else torch.no_grad():
        # with torch.no_grad():
            # 馈送数据
            for data_dict in self.tqdm_loader:
                self.step += 1  # 当前批次数
                self.tqdm_loader.set_description(('Train' if self.tqdm_loader.load_mode == 0 else 'Val') + '-Epoch{}-Step{}'.format(
                    self.epoch, self.step))  # 设置tqdm左边显示内容
                # 优化器梯度清空
                self.zerograd_optimizers()  # 清除梯度值
                # 数据拷贝到device中并输入网络
                self.feed_network(data_dict)
                # 计算评估
                self.calculate_eval()
                # 反向传播
                if self.tqdm_loader.load_mode == 0:
                    self.backward()
                # 进度条右侧更新损失和精度
                self.update_info()

        # 求损失均值
        self.eval_mean()
        # 记录损失均值并写入log
        self.eval_mean_record()
        # 打印在显示台
        print('Epoch:{}  '.format(self.epoch + 1) + str(self.epoch_acc_loss) + str(time_t))

    def feed_network(self, data_dict):
        pass

    def eval_mean_record(self):
        # 记录
        for key, value in self.epoch_acc_loss.dicts.items():
            if not key in self.record_acc_loss.dicts['train' if self.tqdm_loader.load_mode == 0 else 'val'].keys():
                self.record_acc_loss.addKey(key)
            self.record_acc_loss.dicts['train' if self.tqdm_loader.load_mode == 0 else 'val'][key].update({self.epoch: value})

        # 在log中写入轮次epoch评估值
        self.write_losses_SumWri('train' if self.tqdm_loader.load_mode == 0 else 'val')

    def eval_mean(self):
        # 求损失均值
        for key, value in self.epoch_acc_loss.dicts.items():
            self.epoch_acc_loss.dicts[key] = value / self.tqdm_loader.len

    def save_ckpt(self, flag='', save_best=False, save_epoch=None, save_per_epoch=0):
        # ckpt字典更新
        self.ckpt = {
            "cfgs": self.cfgs,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            "record_acc_loss": self.record_acc_loss
        }
        # 如果存在着flag则保存时添加flag
        if save_best:  # 如果是保存最好的
            for no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):  # 反向传播的模式中
                if self.best_acc_loss.dicts[mode] > self.epoch_acc_loss.dicts[mode]:
                    save_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT,
                                             self.FLAG + flag + '_' + mode + '_best.pth')  # 保存路径
                    torch.save(self.ckpt, save_path)  # 执行保存
                    print('{}-{} improved from {:.5f} to {:.5f} and save to {}'.format(flag, mode,
                                                                                       self.best_acc_loss.dicts[mode],
                                                                                       self.epoch_acc_loss.dicts[mode],
                                                                                       save_path))  # 界面进行显示
                    self.best_acc_loss.dicts[mode] = self.epoch_acc_loss.dicts[mode]  # 进一步保存
                else:  # 没有提升则进行打印显示
                    print(mode + ' didn\'t improved from {:.5f}.'.format(self.best_acc_loss.dicts[mode]))

        if save_epoch:  # 按epoch保存的情况
            if (self.epoch) in save_epoch:  # 如果档期那epoch在save列表中
                save_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT,
                                         self.FLAG + flag + '_EP{}.pth'.format(self.epoch))  # 保存路径
                torch.save(self.ckpt, save_path)  # 执行保存
                print('Model have saved to {}'.format(save_path))

        if save_per_epoch and not self.epoch == self.cfgs.TRAIN.STOP_EPOCH:  # 按per_epoch保存的情况
            if ((self.epoch) % save_per_epoch) == 0:  # 如果档期那epoch在save列表中
                save_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT,
                                         self.FLAG + flag + '_EP{}.pth'.format(self.epoch))  # 保存路径
                torch.save(self.ckpt, save_path)  # 执行保存
                print('Model have saved to {}'.format(save_path))

        # 最后一轮默认保存
        save_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.WIT, self.FLAG + '_last.pth')
        torch.save(self.ckpt, save_path)
        print('Model have saved to {}'.format(save_path))

    def update_info(self):
        # tqdm更新step loss和acc
        self.tqdm_loader.set_postfix(eval=(' ' + str(self.step_acc_loss)))

    def calculate_eval(self):
        # step损失归零
        self.step_acc_loss.values2zeros()
        # 如果是输出与真值尺寸不一致
        if self.data_out.shape[-1] != self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]].shape[-1]:
            self.data_out = torch.nn.functional.interpolate(self.data_out, size=self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]].shape[2:], mode='nearest', align_corners=None)
        # 判断是不是多损失输出
        if self.cfgs.TRAIN.MUL_LOSS:
            out_num = len(self.data_out)  # 多损失的话，多损失的数量
        else:
            out_num = 1  # 单个损失，输出为1
        # 根据训练参数Loss Acc进行遍历处理
        for mode_no, mode in enumerate(self.cfgs.TRAIN.ARGUMENTS):
            # 真值数据
            if (self.cfgs.MODEL.NUM_CLASSES <= 2):
                truth = np.squeeze(self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]])  # 真值
            else:
                truth = np.squeeze(self.data_in[self.cfgs.DATA.MODES[self.cfgs.DATA.TRUTH_ORDER]]).long()  # 真值
            # 每个预测输出进行评估，计算损失
            for pred_no in range(out_num):
                # 单输出情况
                if out_num == 1:
                    pred = np.squeeze(self.data_out)
                # 多输出情况
                else:
                    pred = np.squeeze(self.data_out[pred_no])
                # step评估结果
                self.step_acc_loss.dicts[mode] += self.criterions[mode](pred, truth) * (self.cfgs.TRAIN.MUL_LOSS_ALPHA[mode_no] if self.cfgs.TRAIN.MUL_LOSS else 1.0)
            # epoch存储step评估结果
            self.epoch_acc_loss.dicts[mode] += float(self.step_acc_loss.dicts[mode].item()) * self.tqdm_loader.batch_size

    def write_losses_SumWri(self, str_flag):
        # 遍历训练参数,并写入log_write
        for argum_no, argum in enumerate(self.epoch_acc_loss.dicts.keys()):
            self.log_write.add_scalar(str_flag + '_' + argum, self.epoch_acc_loss.dicts[argum], self.epoch)


    def backward(self):
        # 确保是训练模式下进行反向传播
        if self.tqdm_loader.load_mode == 0:
            # 反向传播的参数中
            for no, mode in enumerate(self.cfgs.TRAIN.BACKWARD):
                # 反向传播参数乘以相应的缩放值
                self.step_acc_loss.dicts[mode] = (self.step_acc_loss.dicts[mode] * self.cfgs.TRAIN.BACKWARD_WEIGHT[no]).requires_grad_()  # 使其能具有梯度属性
                # amp混合精度的话
                if self.cfgs.TRAIN.IS_AMP:
                    # from apex import amp
                    # with amp.scale_loss(self.step_acc_loss.dicts[mode], self.optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    self.scaler.scale(self.step_acc_loss.dicts[mode]).backward()  # 使用amp后对损失放大进行,梯度反向传播
                    # 带有梯度累积的Loss梯度下降
                    if (self.step % self.cfgs.TRAIN.BACKWARD_ACCUMULATION) == 0:
                        self.scaler.step(self.optimizer)  # # 如果出现了inf或者NaN，scaler.step(optimizer)会忽略此次的权重更新
                        # # 准备着，看是否要增大scaler  如果没有出现inf或者NaN，那么权重正常更新，
                        # # 当连续多次没有出现inf或者NaN，则scaler.update()会将scaler的大小增加
                        self.scaler.update()
                # 正常训练的情况
                else:
                    self.step_acc_loss.dicts[mode].backward()  # 梯度反向传播
                    # 带有梯度累积的Loss梯度下降
                    if (self.step % self.cfgs.TRAIN.BACKWARD_ACCUMULATION) == 0: self.optimizer.step()  # 梯度下降

    def zerograd_optimizers(self):
        # 优化器梯度归零
        self.optimizer.zero_grad()  # 清除梯度值

    def record_plot(self):
        '''
        '''
        styles = ['-.o', '-^', '-.+', '-s', '-.H', '-d', '-.>', '-<', '-.3', '-*', '-.p', '-v', '-o', '-.^', '-+', '-.s', '-H', '-.d', '->', '-.<', '-3', '-.*', '-p', '-.v']
        colors = ['r', 'g', 'b', 'm', 'k', 'c', 'y', 'r', 'g', 'b', 'm', 'k', 'c', 'y', 'r', 'g', 'b', 'm', 'k', 'c', 'y']
        subplotxth, subplotyth = len(self.record_acc_loss.dicts['train']), 1
        plt.figure(num=0, figsize=(subplotxth * 9, subplotyth * 6), dpi=100)

        for arg_no, arg_mode in enumerate(self.record_acc_loss.dicts['train']):
            for tar_no, tra_mode in enumerate(self.record_acc_loss.dicts.keys()):
                plt.subplot(subplotyth, subplotxth, arg_no + 1)
                plt.plot(list(self.record_acc_loss.dicts[tra_mode][arg_mode].keys()),
                         list(self.record_acc_loss.dicts[tra_mode][arg_mode].values()),
                         colors[arg_no * len(self.record_acc_loss.dicts) + tar_no] + styles[arg_no * len(self.record_acc_loss.dicts) + tar_no],
                         label=tra_mode + '-' + arg_mode)
            plt.title(arg_mode)
            plt.legend()

        now_time = time.strftime('%Y{y}%m{m}%d{d}-%H{h}%M{M}%S{s}').format(y='年', m='月', d='日', h='时', M='分', s='秒')
        img_path = os.path.join(self.cfgs.DATA.RESULT_DIR, self.cfgs.DATA.ACCLOSS, self.FLAG + now_time)
        img_path = img_path + '.tif'
        plt.savefig(img_path)

    def add_loss_key(self, mode, add_key):
        # 添加相应的key字段 Seg
        self.step_acc_loss.addKey(mode + add_key)
        self.epoch_acc_loss.addKey(mode + add_key)
        self.record_acc_loss.addKey(mode + add_key)

if __name__ == '__main__':
    pass

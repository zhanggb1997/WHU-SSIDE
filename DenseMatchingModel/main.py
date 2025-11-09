'''
Project    : RSDetec
FileName   : main_dm .py
CreateTime : 2023/8/25 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import os
import random
import numpy
import torch
import argparse
from configs.DenseMatch import CN_, update_configs
from core.DM_core.DM_test import TestDM
from core.DM_core.DM_train import TrainDM
from models.DM_models.Meta_MRGE.Meta_MRGE import MetaMRGE
from utils.log import logS


#################  配置超参数  #########################
parser = argparse.ArgumentParser()
parser.add_argument("--cfgs", type=str, default=[
    # data
    r'./configs/DenseMatch/data/Our.yaml',
    # r'./configs/DenseMatch/data/GF7.yaml',
    # model
    # r'./configs/DenseMatch/model/Meta-MRGE.yaml',
    # r'./configs/DenseMatch/model/IGEV.yaml',
    # r'./configs/DenseMatch/model/HMSMNetOri.yaml',
    # r'./configs/DenseMatch/model/CSTR.yaml',
    # r'./configs/DenseMatch/model/STTR.yaml',
    # r'./configs/DenseMatch/model/PCWNet.yaml',
    # r'./configs/DenseMatch/model/PSMNet.yaml',
    # r'./configs/DenseMatch/model/HD3Net.yaml',
    # r'./configs/DenseMatch/model/FADNet.yaml',
    # train
    r'./configs/DenseMatch/train/Train_base.yaml',
    # r'./configs/DenseMatch/train/Pred_base.yaml',
], help="Paths of programme configure files")

args = parser.parse_args()
# 打印args信息
print(args.cfgs)
# 将配置文件载入进行更新
update_configs(CN_=CN_, cfgs_path=args.cfgs)
#################  环境配置  #########################
# 随机种子设定
seed = CN_.SEED  # 随机种子设定
torch.manual_seed(seed)  # 固定随机种子
torch.cuda.manual_seed(seed)  # 固定随机种子
torch.cuda.manual_seed_all(seed)  # 固定随机种子
numpy.random.seed(seed)  # 固定随机种子
random.seed(seed)
# cuDNN设置
CUDNN = CN_.CUDNN
torch.backends.cudnn.determinstic = CUDNN.DETERMINSTIC  # 保证可重复性
torch.backends.cudnn.enabled = CUDNN.ENABLED  # 提升计算速率
torch.backends.cudnn.benchmark = CUDNN.BENCHMARK  # 提升计算速率

# # 多GPU设定
# torch.cuda.set_device(args.local_rank)
# 调用torch.distributed下任何函数前，必须运行torch.distributed.init_process_group(backend='nccl')初始化
# 1.pytorch支持的通讯后端，2.各级器间的通讯方式，单机就是localhost
# 3.rank标识主机和从机，只有一个主机则设定为0，4.world_size标识多少主机，只有一个主机设定为1
# torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

# #################  输出Flag  #########################
if CN_.GPUS == torch.cuda.device_count():
    logS.info("成功检测到 {} 个GPU！".format(CN_.GPUS))
else:
    logS.warning("没有检测到GPU/GPU设定数量不对！")

#################  配置模型  #########################
# model = HMSMNetOri(CN_)
# model = CSTR(CN_)
# model = STTR(CN_)
# model = PCWNet(CN_, True)
# model = PSMNet(CN_)
# model = HD3Net(CN_)
# model = FADNet(CN_)
# model = IGEVStereo(CN_)
model = MetaMRGE(CN_)
# stat(model, (1, 1024, 1024))

#################  开始训练-测试预测  #########################
if ('train' in CN_.MODE) or ('Train' in CN_.MODE):
    # 训练
    train_env = TrainDM(CN_, model)
    train_env.train_model()
    # 预测
    test_env = TestDM(CN_, model)
    test_env.test_model()

#################  只进行预测  #########################
elif ('pred' in CN_.MODE) or ('Pred' in CN_.MODE):
    assert os.path.exists(CN_.RESUME.CKPT_DIR), "您输入的预训练权重不存在！"
    # 加载权重
    test_env = TestDM(CN_, model)
    test_env.test_model()

#################  只进行评估  #########################
elif ('calcul' in CN_.MODE) or ('Calcul' in CN_.MODE):
    # 加载权重
    test_env = TestDM(CN_, model)
    test_env.calculate_precise()


if __name__ == '__main__':
    pass

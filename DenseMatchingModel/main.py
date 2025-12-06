'''
Project    : RSDetec
FileName   : main_dm .py
CreateTime : 2023/8/25 
=======================
@CopyRight : WHU
@Author    : Zhang
@Contact   : zhanggb1997@163.com
@Content   : # Implementation Content #
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

# =========================================================================
# 1. Hyperparameter Configuration
# =========================================================================
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
print(args.cfgs)
update_configs(CN_=CN_, cfgs_path=args.cfgs)

# =========================================================================
# 2. Environment Configuration
# =========================================================================
seed = CN_.SEED
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
random.seed(seed)

CUDNN = CN_.CUDNN
torch.backends.cudnn.determinstic = CUDNN.DETERMINSTIC
torch.backends.cudnn.enabled = CUDNN.ENABLED
torch.backends.cudnn.benchmark = CUDNN.BENCHMARK

# =========================================================================
# 3. Model Configuration
# =========================================================================
# model = HMSMNet(CN_)
# model = CSTR(CN_)
# model = STTR(CN_)
# model = PCWNet(CN_, True)
# model = PSMNet(CN_)
# model = HD3Net(CN_)
# model = FADNet(CN_)
# model = IGEVStereo(CN_)
model = MetaMRGE(CN_)
# stat(model, (1, 1024, 1024))

# =========================================================================
# 4. Execution Logic (Train / Test / Predict)
# =========================================================================
if ('train' in CN_.MODE) or ('Train' in CN_.MODE):
    train_env = TrainDM(CN_, model)
    train_env.train_model()
    test_env = TestDM(CN_, model)
    test_env.test_model()

elif ('pred' in CN_.MODE) or ('Pred' in CN_.MODE):
    assert os.path.exists(CN_.RESUME.CKPT_DIR), "The provided pretrained weight path does not exist!"
    test_env = TestDM(CN_, model)
    test_env.test_model()

elif ('calcul' in CN_.MODE) or ('Calcul' in CN_.MODE):
    test_env = TestDM(CN_, model)
    test_env.calculate_precise()


if __name__ == '__main__':
    pass

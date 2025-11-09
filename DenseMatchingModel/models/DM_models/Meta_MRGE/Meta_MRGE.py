import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DM_models.OurNet6_IGEV1.update import BasicMultiUpdateBlock
from models.DM_models.OurNet6_IGEV1.extractor import MultiBasicEncoder, Feature
from models.DM_models.OurNet6_IGEV1.geometry import Combined_Geo_Encoding_Volume
from models.DM_models.OurNet6_IGEV1.submodule import *
from models.DM_models.OurNet6_IGEV1.FeatExtract import FeatExtract
from models.DM_models.OurNet6_IGEV1.FeatFusion import FeatFusion
from models.DM_models.OurNet6_IGEV1.CostUtils import CostConcatenation, GWCCostConcat, Hourglass, FeatureFusion, \
    Estimation, Refinement, DispRegress

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglassgwc(nn.Module):
    def __init__(self, in_channels):
        super(hourglassgwc, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*8, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*8, in_channels*4, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, IN=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.feature_att_4 = FeatureAtt(in_channels, 48)
        self.feature_att_8 = FeatureAtt(in_channels*2, 96)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*8, 384)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 96)

    def forward(self, x, features):
        conv0 = self.conv0(x)
        conv0 = self.feature_att_4(conv0, features[0])

        conv1 = self.conv1(conv0)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class hourglass_d2(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_d2, self).__init__()

        self.conv0 = BasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*8, in_channels*8, is_3d=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))


        self.conv3_up = BasicConv(in_channels*8, in_channels*4, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True,
                                  relu=True, kernel_size=(1, 1, 1), padding=(1, 1, 1), stride=(1, 1, 1))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, IN=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.feature_att_4 = FeatureAtt(in_channels, 96)
        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*8, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv0 = self.conv0(x)
        conv0 = self.feature_att_4(conv0, features[0])

        conv1 = self.conv1(conv0)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class MetaMRGE(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs

        # 最大最小视差
        self.max_disp = int(cfgs.TRAIN.MAX_DISP)
        self.min_disp = int(cfgs.TRAIN.MIN_DISP)

        # GRU迭代次数
        self.iters = int(cfgs.MODEL.SET.IGEV.GRU_ITERS)

        # 大中小视差范围
        self.l_min_disp = int(cfgs.TRAIN.MIN_DISP)
        self.l_max_disp = int(cfgs.TRAIN.MAX_DISP)

        self.m_min_disp = self.cfgs.TRAIN.M_MIN_DISP  # -48 / +48
        self.m_max_disp = self.cfgs.TRAIN.M_MAX_DISP
        self.m_min_disp_ = (self.m_min_disp - self.min_disp)  # 80 / 176
        self.m_max_disp_ = (self.m_max_disp - self.min_disp)

        self.s_min_disp = self.cfgs.TRAIN.S_MIN_DISP  # -24 / +24
        self.s_max_disp = self.cfgs.TRAIN.S_MAX_DISP
        self.s_min_disp_ = (self.s_min_disp - self.min_disp)  # 104 / 152
        self.s_max_disp_ = (self.s_max_disp - self.min_disp)

        # context上下文通道数
        self.context_dims = cfgs.MODEL.SET.IGEV.HIDDEN_DIMS
        self.hidden_dims = cfgs.MODEL.SET.IGEV.HIDDEN_DIMS


        # ========= 特征提取 ==========
        # 基础特征提取
        self.featext = FeatExtract(cfgs, False)
        # 多尺度特征融合
        self.featfuse = FeatFusion(cfgs)
        # self.feature = Feature()


        # ========= 代价构建 ==========
        # # CNN代价构建
        # self.cost16 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        # self.cost8 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        # self.cost4 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        # gwc级联代价体构建
        self.GWCcost4 = GWCCostConcat(min_disp=self.l_min_disp // self.cfgs.TRAIN.INTER_DISP,  max_disp=self.l_max_disp // self.cfgs.TRAIN.INTER_DISP, num_groups=8)
        self.patch0 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False)
        self.patch1 = nn.Conv3d(8, 8, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=False)

        # ========= 代价聚合 ==========
        # # CNN代价聚合
        # self.aggre16 = Hourglass(filters=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.aggre8_1 = Hourglass(filters=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.aggre8 = Hourglass(filters=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.aggre4_1 = Hourglass(filters=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.aggre4 = Hourglass(filters=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # # CNN代价融合
        # self.cfuse8 = FeatureFusion(units=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM, infeatures=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.cfuse4 = FeatureFusion(units=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM, infeatures=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # gwc代价聚合
        self.agg4_gwc_s = hourglassgwc(8)
        self.agg4_gwc_m = hourglassgwc(8)
        self.agg4_gwc_l = hourglassgwc(8)

        # ========= 视差计算 ==========
        # self.estim16 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16, input_channels=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.estim8 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8, input_channels=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.estim4 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4, input_channels=cfgs.MODEL.SET.BM.COSTFUSION.FILTER_NUM)
        # self.refiner = Refinement(filters=64)

        self.estim4_gwc_s = Estimation(min_disp=self.s_min_disp//self.cfgs.TRAIN.SCALE_DISP, max_disp=self.s_max_disp//self.cfgs.TRAIN.SCALE_DISP,  interval=self.cfgs.TRAIN.INTER_DISP  //self.cfgs.TRAIN.SCALE_DISP, input_channels=8)
        self.estim4_gwc_m = Estimation(min_disp=self.m_min_disp//self.cfgs.TRAIN.SCALE_DISP, max_disp=self.m_max_disp//self.cfgs.TRAIN.SCALE_DISP,  interval=self.cfgs.TRAIN.INTER_DISP*2//self.cfgs.TRAIN.SCALE_DISP, input_channels=8)
        self.estim4_gwc_l = Estimation(min_disp=self.l_min_disp//self.cfgs.TRAIN.SCALE_DISP, max_disp=self.l_max_disp//self.cfgs.TRAIN.SCALE_DISP,  interval=self.cfgs.TRAIN.INTER_DISP*4//self.cfgs.TRAIN.SCALE_DISP, input_channels=8)
        # self.estim4_gwc_s = DispRegress(mindisp=self.s_min_disp // 4, maxdisp=self.s_max_disp // 4, input_channels=8)
        # self.estim4_gwc_m = DispRegress(mindisp=self.m_min_disp // 8, maxdisp=self.m_max_disp // 8, input_channels=8)
        # self.estim4_gwc_l = DispRegress(mindisp=self.l_min_disp //16, maxdisp=self.l_max_disp //16, input_channels=8)


        # # =========  GRU  ==========
        self.conv = BasicConv(80, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.disp_conv = nn.Sequential(
            BasicConv(3, 64, kernel_size=1, stride=1, padding=0),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.selective_conv = nn.Sequential(
            BasicConv(48 + 64, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 3, 3, 1, 1, bias=False),
        )
        # # 多层次编码层
        self.cnet = MultiBasicEncoder(cfgs, output_dim=[self.hidden_dims, self.context_dims], norm_fn="batch", downsample=cfgs.MODEL.SET.IGEV.N_DOWNSAMPLE)
        # # GRU 层迭代
        self.update_block = BasicMultiUpdateBlock(self.cfgs, hidden_dims=self.hidden_dims)
        # # 上下文信息卷积层
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(self.context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2) for i in range(self.cfgs.MODEL.SET.IGEV.N_GRU_LAYERS)])

        self.spx_2_gru = Conv2x(64, 16, True)
        # self.spx_gru = nn.Sequential(nn.ConvTranspose2d(32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(32, 9, kernel_size=8, stride=4, padding=2), )
        self.spx_gru_lxy = nn.Sequential(
            BasicConv(9+3, 16, kernel_size=3, stride=1, padding=1),
            # BasicConv(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0), )

        self.spx = nn.Sequential(nn.ConvTranspose2d(32, 9, kernel_size=8, stride=4, padding=2), )
        self.spx_2 = Conv2x(64, 16, True)
        self.spx_4 = nn.Sequential(
            BasicConv(48, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
        )


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x, disp_scale=4., rgb=None, dx=None, dy=None):
        # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
        with autocast(enabled=True, dtype=getattr(torch, 'float16', torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            if rgb is not None:
                spx_pred = self.spx_gru_lxy(torch.cat((spx_pred, rgb, dx, dy), dim=1))
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*disp_scale, spx_pred)
        return up_disp

    def forward(self, data):
        """ Estimate disparity between pair of frames """

        # ============  输入信息处理  ===========
        # 输入立体图像
        imageL = data['x_l']
        imageR = data['x_r']

        # meta 信息
        if self.cfgs.DATA.ADD_META:
            metaL = data['meta']
            metaR = data['meta']
        else:
            metaL = None
            metaR = None

        # ============  基础特征提取融合  ===========
        # 特征提取与融合
        # [B, C=384, H/32, W/32], [B, C=192, H/32, W/16], [B, C=96, H/8, W/8], [B, C=48, H/4, W/4], [B, C, H/1, W/1]
        featL = self.featext(imageL, None, metaL, "left")
        featR = self.featext(imageR, None, metaR, "right")

        ffuseL = self.featfuse(featL)
        ffuseR = self.featfuse(featR)

        # [B, C=16, H/16, W/16], [B, C=16, H/8, W/8], [B, C=16, H/4, W/4], [B, C=16, H/4, W/4]
        featL16, featL8, featL4, featL4_1 = ffuseL
        featR16, featR8, featR4, featR4_1 = ffuseR

        # # ============  代价构建  ===========
        # # gwc级联代价体构建
        featL4_cat = torch.cat((featL[3], featL4, featL4_1), dim=1)  # [B, C=48+16+16=80, H/4, W/4] [B, C=48+32+32=112, H/4, W/4]
        featR4_cat = torch.cat((featR[3], featR4, featR4_1), dim=1)  # [B, C=48+16+16=80, H/4, W/4] [B, C=48+32+32=112, H/4, W/4]
        featL4_gwc = self.desc(self.conv(featL4_cat))  # [B, C=96, H/4, W/4]
        featR4_gwc = self.desc(self.conv(featR4_cat))  # [B, C=96, H/4, W/4]
        costvol4_gwc = self.GWCcost4(featL4_gwc, featR4_gwc)  # [B, C=8, D=48, H/4, W/4]

        costvol4_gwc_s = costvol4_gwc[:, :, self.s_min_disp_//self.cfgs.TRAIN.INTER_DISP:self.s_max_disp_//self.cfgs.TRAIN.INTER_DISP]  # [N, C=8, D=12, H/4, W/4]
        costvol4_gwc_m = self.patch0(costvol4_gwc[:, :, self.m_min_disp_//self.cfgs.TRAIN.INTER_DISP:self.m_max_disp_//self.cfgs.TRAIN.INTER_DISP])  # [N, C=8, D=24->12, H/4, W/4]
        costvol4_gwc_l = self.patch1(costvol4_gwc)  # [N, C=8, D=48->12, H/4, W/4]


        # ============  代价聚合  ===========
        # gwc级联代价聚合
        agg_cost4_gwc_s = self.agg4_gwc_s(costvol4_gwc_s, featL[:4][::-1])
        agg_cost4_gwc_m = self.agg4_gwc_m(costvol4_gwc_m, featL[:4][::-1])
        agg_cost4_gwc_l = self.agg4_gwc_l(costvol4_gwc_l, featL[:4][::-1])

        # # ============  代价计算与细化  ============
        disp4_s = self.estim4_gwc_s(agg_cost4_gwc_s)
        disp4_m = self.estim4_gwc_m(agg_cost4_gwc_m)
        disp4_l = self.estim4_gwc_l(agg_cost4_gwc_l)

        # ============  GRU  ============
        disp_feature = self.disp_conv(torch.cat([disp4_s, disp4_m, disp4_l], dim=1))  # [N, C=64, H/4, W/4]
        selective_weights = torch.sigmoid(self.selective_conv(torch.cat([featL[3], disp_feature], dim=1)))  # [N, C=160->128, H/4, W/4]

        # [N, C=128, H/4, W/4]  [N, C=128, H/8, W/8]  [N, C=128, H/16, W/16]
        cnet_list = self.cnet(imageL, num_layers=self.cfgs.MODEL.SET.IGEV.N_GRU_LAYERS)
        net_list = [torch.tanh(x[0]) for x in cnet_list]  # tanh处理0
        inp_list = [torch.relu(x[1]) for x in cnet_list]  # relu处理1
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

        # # geo encode
        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(agg_cost4_gwc_s.float(), agg_cost4_gwc_m.float(), agg_cost4_gwc_l.float(), featL4_gwc.float(), featR4_gwc.float(), radius=4)
        b, c, h, w = featL4_gwc.shape
        coords = torch.arange(w).float().to(featL4_gwc.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = disp4_s
        iter_preds = []

        # GRUs iterations to update disparity GRU更新视差图
        for itr in range(self.iters):
            disp = disp.detach()
            geo_feat0, geo_feat1, geo_feat2, init_corr = geo_fn(disp, coords)
            # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            with autocast(enabled=True, dtype=getattr(torch, 'float16', torch.float16)):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat0, geo_feat1, geo_feat2, init_corr, selective_weights, disp, self.cfgs.MODEL.SET.IGEV.N_GRU_LAYERS==3, iter08=self.cfgs.MODEL.SET.IGEV.N_GRU_LAYERS>=2)

            disp = disp + delta_disp
            if (('pred' in self.cfgs.MODE) or ('Pred' in self.cfgs.MODE)) and itr < self.iters-1:
                continue

            # upsample predictions 上采样视差预测结果
            disp_up = self.upsample_disp(disp, mask_feat_4, featL4_1, self.cfgs.TRAIN.SCALE_DISP, data['x_l'], data['dx'], data['dy'])
            iter_preds.append(disp_up)

        if (('pred' in self.cfgs.MODE) or ('Pred' in self.cfgs.MODE)):
            return disp_up

        # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
        with autocast(enabled=True, dtype=getattr(torch, 'float16', torch.float16)):
            xspx = self.spx_4(featL[3])
            xspx = self.spx_2(xspx, featL4_1)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)
        agg_disp0 = context_upsample(disp4_s*self.cfgs.TRAIN.SCALE_DISP, spx_pred.float())
        agg_disp1 = context_upsample(disp4_m*self.cfgs.TRAIN.SCALE_DISP, spx_pred.float())
        agg_disp2 = context_upsample(disp4_l*self.cfgs.TRAIN.SCALE_DISP, spx_pred.float())

        # agg预测结果， iter预测结果
        return [agg_disp0, agg_disp1, agg_disp2], iter_preds

'''
Project    : RSDetec
FileName   : DispRegress .py
CreateTime : 2024/10/10 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from models.DM_models.CSTR.misc import batched_index_select, torch_1d_sample, NestedTensor



class ContextAdjustmentLayer(nn.Module):
    """
    Adjust the disp and occ based on image context, design loosely follows https://github.com/JiahuiYu/wdsr_ntire2018
    """

    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(1 + 1, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

        # occ head
        self.occ_head = nn.Sequential(
            weight_norm(nn.Conv2d(1 + 1, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, disp_raw: Tensor, occ_raw: Tensor, img: Tensor):
        """
        :param disp_raw: raw disparity, [N,1,H,W]
        :param occ_raw: raw occlusion mask, [N,1,H,W]
        :param img: input left image, [N,3,H,W]
        :return:
            disp_final: final disparity [N,1,H,W]
            occ_final: final occlusion [N,1,H,W] 
        """""
        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:  # 遍历8次resnet
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res

        occ_final = self.occ_head(torch.cat([occ_raw, img], dim=1))

        return disp_final, occ_final


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale


def build_context_adjustment_layer(args):
    if args.MODEL.SET.BM.DISPREGRESSION.CONTEXT_ADJUST_LAYER == 'cal':
        return ContextAdjustmentLayer(args.MODEL.SET.BM.DISPREGRESSION.CAL_NUM_BLOCKS,
                                      args.MODEL.SET.BM.DISPREGRESSION.CAL_FEAT_DIM,
                                      args.MODEL.SET.BM.DISPREGRESSION.CAL_EXPANSION_RATIO)
    elif args.context_adjustment_layer == 'none':
        return None
    else:
        raise ValueError(f'Context adjustment layer option not recognized: {args.context_adjustment_layer}')


class RegressionHead(nn.Module):
    """
    Regress disparity and occlusion mask 回归视差和遮挡mask
    """

    def __init__(self, cal: nn.Module, ot: bool = True):
        super(RegressionHead, self).__init__()
        self.cal = cal
        self.ot = ot
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost
        # self.masknet = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 64 * 9, 1, padding=0))
        # self.masknet = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 16 * 9, 1, padding=0))

    def _compute_unscaled_pos_shift(self, w: int, device: torch.device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate
        disparity
        计算从左图像到右图像的每个像素位置之间的相对差，用于计算视差。
        :param w: image width 图像宽度W
        :param device: torch device 设备
        :return: relative pos shifts 相对位置偏移
        """
        pos_r = torch.linspace(0, w - 1, w)[None, None, None, :].to(device)  # 1 x 1 x 1 x W_right
        pos_l = torch.linspace(0, w - 1, w)[None, None, :, None].to(device)  # 1 x 1 x W_left x1
        pos = pos_l - pos_r
        # pos[pos < 0] = 0
        return pos

    def _compute_low_res_disp(self, pos_shift: Tensor, attn_weight: Tensor, occ_mask=None):
        """
        Compute low res disparity using the attention weight by finding the most attended pixel and regress within the 3px window
        通过找到关注度最高的像素并在3px窗口内回归，使用注意力权重计算低分辨率视差
        :param pos_shift: relative pos shift (computed from _compute_unscaled_pos_shift), [1,1,W,W]  位置偏移
        :param attn_weight: attention (computed from _optimal_transport), [N,H,W,W]  attn_ot 最优传输得到的attn
        :param occ_mask: ground truth occlusion mask, [N,H,W]  遮挡mask
        :return: low res disparity, [N,H,W] and attended similarity sum, [N,H,W]
        """

        # find high response area  返回指定维度最大值的序号
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW   [N, H/4, W/4, W'/4] ==> [N, H/4, W/4]

        # build 3 px local window 最大值序号周围的3像素窗口
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)  # NxHxWx3  [N, H/4, W/4, 3]

        # attention with re-weighting 3像素窗口内获取特征, 重参数化的权重
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)  # N x Hx W_left x (W_right+2)
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)  # offset range by 1, N x H x W_left x 3

        # compute sum of attention 加和3像素窗口中的attn
        norm = attn_weight_rw.sum(-1, keepdim=True)  # [N, H/4, W/4, 1]
        if occ_mask is None:
            norm[norm < 0.051] = 1.0
        else:
            norm[occ_mask, :] = 1.0  # set occluded region norm to be 1.0 to avoid division by 0 将遮挡区域范数设置为1.0以避免被0分割

        # re-normalize to 1, 重参数化的位置偏移
        attn_weight_rw = attn_weight_rw / norm  # re-sum to 1 保证加起来等于1
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad)
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)

        # compute low res disparity 计算视差
        disp_pred_low_res = (attn_weight_rw * pos_rw)  # NxHxW



        # # 可能性/概率
        # probabilities = torch.softmax(attn_weight, -1)
        # # 最终视差
        # disp_pred_low_res = pos_shift * probabilities
        # # 最终概率
        # norm = torch.sum(probabilities ** 2, -1)


        return disp_pred_low_res.sum(-1), norm


    def _compute_gt_location(self, scale: float, sampled_cols: Tensor, sampled_rows: Tensor,
                             attn_weight: Tensor, disp: Tensor):
        """
        Find target locations using ground truth disparity.
        Find ground truth response at those locations using attention weight.
        使用真实视差查找目标位置。
        使用注意力权重（预测值）找出这些位置的真值的反应。
        :param scale: high-res to low-res disparity scale 高分辨率到低分辨率视差倍数
        :param attn_weight: attention weight (output from _optimal_transport), [N, H', W'l, W'r] 注意力权重（optimal_transport的输出）[N, H', W'l', W'r]
        :param disp: ground truth disparity 视差真值 [N, C, H, W]
        :return: response at ground truth location [N,H,W,1] and target ground truth locations [N,H,W,1] 真值位置反馈， 目标真值位置
        """
        disp = torch.squeeze(disp, 1)
        # compute target location at full res  以全分辨率计算目标位置
        _, _, w = disp.size()  # [N, H, Wl]
        pos_l = torch.linspace(0, w - 1, w)[None,].to(disp.device)  # [Wl, 1] [0 ~ W-1]
        # flag
        disp_ = deepcopy(disp)
        disp_[disp <= -999] = 0
        target = (pos_l - disp_)[..., None]  # [N, H, Wl, 1]    pos_L-disp=pos_R == 真值，Left图对应的right图的位置是多少

        # sampled cols rows
        if sampled_cols is not None:
            target = batched_index_select(target, 2, sampled_cols)
        if sampled_rows is not None:
            target = batched_index_select(target, 1, sampled_rows)
        target = target / scale  # scale target location 缩放

        # compute ground truth response location for rr loss 计算rr损失的真值响应位置,attn_weight中W/4与W'/4层面中的对应位置
        # 根据right位置floor_ceil，计算linear权重，取出attn中的对应左右向量，结合权重进行计算真实视差响应区域
        gt_response = torch_1d_sample(attn_weight, target, 'linear')  # [N, H', W'l]

        mask = torch.logical_and((target > 0), (target <= (w/scale - 1)))

        return gt_response, target, mask  # [N, H', W'l]   [N, H', W'l, 1]

    def _upsample(self, x_l: NestedTensor, disp_pred: Tensor, occ_pred: Tensor, scale: float):
        """
        Upsample the raw prediction to full resolution
        将原始预测上采样到全分辨率
        :param x: input data
        :param disp_pred: predicted disp at low res
        :param occ_pred: predicted occlusion at low res  [N, C=16*9=144, H/4, W/4]
        :param mask:  [N, C=16*9=144, H/4, W/4]
        :param scale: high-res to low-res disparity scale
        :return: high res disp and occ prediction
        """
        _, _, h, w = x_l.size()
        # N,H,W=disp_pred.size()
        # scale disparity
        disp_pred_attn = disp_pred * scale  # [N, H/4, W/4]
        # plt.figure(figsize=(8, 6))
        # plt.imshow(disp_pred.cpu().squeeze().numpy())

        # #gwy_upsample*4
        # mask = mask.view(N, 1, 9, 4, 4, H, W)  # [N, 1, 9, 4, 4, H/4, W/4]
        # mask = torch.softmax(mask, dim=2)  # [N, 1, 9, 4, 4, H/4, W/4]
        # # unfold的作用就是手动实现(卷积中)的滑动窗口操作，也就是只有卷，没有积 [N, 1, W/4, W/4] ==> [N, 1*ks_w*ks_h, W/4*W/4]
        # up_flow = F.unfold(disp_pred_attn[:,None,], [3,3], padding=1)
        #
        # up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)  # [N, ks_w, ks_h, W/4, W/4]
        #
        # up_flow = torch.sum(mask * up_flow, dim=2)  # [N, 1, ks_w * ks_h, 1, 1, W/4, W/4] ==> [N, 1, ks_w * ks_h, 4, 4, W/4, W/4] ==> [N, 1, 4, 4, W/4, W/4]
        # up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [N, 1, 4, 4, W/4, W/4] ==> [N, 1, W/4, 4, W/4, 4]
        # disp_pred=up_flow.reshape(N,1, 4*H, 4*W)  # [N, 1, W, W]

        disp_pred = F.interpolate(disp_pred_attn[:, None, ], size=(h, w), mode='nearest')
        # disp_pred = F.interpolate(disp_pred_attn[:, None, ], size=(h, w), mode='bilinear')
        occ_pred = F.interpolate(occ_pred[:, None, ], size=(h, w), mode='nearest')  # N x 1 x H x W
        if self.cal is not None:
            # normalize disparity 标准化视差
            eps = 1e-6
            mean_disp_pred = disp_pred.mean()
            std_disp_pred = disp_pred.std() + eps
            disp_pred_normalized = (disp_pred - mean_disp_pred) / std_disp_pred

            # normalize occlusion mask 标准化遮挡
            occ_pred_normalized = (occ_pred - 0.5) / 0.5
            # resblock处理
            disp_pred_normalized, occ_pred = self.cal(disp_pred_normalized, occ_pred_normalized, x_l)  # N x H x W
            # 反算回去，得到最终的disp
            disp_pred_final = disp_pred_normalized * std_disp_pred + mean_disp_pred
        else:
            disp_pred_final = disp_pred.squeeze(1)
            disp_pred_attn = disp_pred_attn.squeeze(1)

        return disp_pred_final.squeeze(1), disp_pred.squeeze(1), disp_pred_attn, occ_pred.squeeze(1)

    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int):
        """
        Sinkhorn Normalization in Log-space as matrix scaling problem.
        Regularization strength is set to 1 to avoid manual checking for numerical issues
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)
        对数空间中的Sinkhorn归一化作为矩阵缩放问题。
        正则化强度设置为1，以避免手动检查数值问题
        :param attn: input attention weight, [N,H,W+1,W+1]
        :param log_mu: marginal distribution of left image, [N,H,W+1]
        :param log_nu: marginal distribution of right image, [N,H,W+1]
        :param iters: number of iterations
        :return: updated attention weight
        """

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for idx in range(iters):
            # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)

        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor, iters: int):
        """
        Perform Differentiable Optimal Transport in Log-space for stability
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)
        在对数空间中执行可微分最优传输以获得稳定性
        改编自 SuperGlue
        :param attn: raw attention weight, [N,H,W,W]  原始注意力权重 [N, H/4, W/4, W'/4]
        :param iters: number of iterations to run sinkhorn  运行sinkhorn的迭代次数
        :return: updated attention weight, [N,H,W+1,W+1]  更新注意力权重
        """
        bs, h, w, _ = attn.shape  # [N, H', W'l, W'r]

        # set marginal to be uniform distribution  将边际设为均匀分布
        marginal = torch.cat([torch.ones([w]), torch.tensor([w]).float()]) / (2 * w)  # [W'l' + 1]
        log_mu = marginal.log().to(attn.device).expand(bs, h, w + 1)  # [N, H', W'l+1]
        log_nu = marginal.log().to(attn.device).expand(bs, h, w + 1)  # [N, H', W'l+1]

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)  # [N, H', W'l, W'r+1]
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)  # [N, H', W'l+1, W'r+1]

        # sinkhorn, 遍历十次 对数空间中的Sinkhorn归一化作为矩阵缩放问题。正则化强度设置为1，以避免手动检查数值问题
        attn_ot = self._sinkhorn(similarity_matrix, log_mu, log_nu, iters)  # [N, H', W'l+1, W'r+1]

        # convert back from log space, recover probabilities by normalization 2W  从log空间转换回来，通过归一化2W恢复概率
        attn_ot = (attn_ot + torch.log(torch.tensor([2.0 * w]).to(attn.device))).exp()  # [N, H', W'l+1, W'r+1]

        return attn_ot

    def _softmax(self, attn: Tensor):
        """
        Alternative to optimal transport

        :param attn: raw attention weight, [N,H,W,W]
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        attn_softmax = F.softmax(similarity_matrix, dim=-1)

        return attn_softmax

    def _compute_low_res_occ(self, matched_attn: Tensor):
        """
        Compute low res occlusion by using inverse of the matched values
        使用匹配值的倒数计算低分辨率遮挡
        :param matched_attn: updated attention weight without dustbins, [N,H,W,W]  [N, H/4, W/4, 1]
        :return: low res occlusion map, [N,H,W]
        """
        occ_pred = 1.0 - matched_attn
        return occ_pred.squeeze(-1)

    def forward(self, attn_weight: Tensor, x_l: Tensor, real_disp=None, sampled_cols=None, sampled_rows=None):
        """
        Regression head follows steps of
            - compute scale for disparity (if there is downsampling)
            - impose uniqueness constraint by optimal transport
            - compute RR loss
            - regress disparity and occlusion
            - upsample (if there is downsampling) and adjust based on context
        回归头遵循以下步骤
            - 计算视差的比例（如果存在下采样）
            - 最优运输的唯一性约束
            - 计算RR损失
            - 回归视差和遮挡
            - 上采样（如果存在下采样）并根据上下文进行调整

        :param attn_weight: raw attention weight, [N,H',W'l,W'r]
        :param x: input data
        :return: dictionary of predicted values  预测结果值字典
        """
        output = {}  # 输出

        # compute scale 计算缩放比
        scale = x_l.shape[-1] // attn_weight.shape[-1]  # 根据尺寸计算所得的缩放系数

        # normalize attention to 0-1  标准化attn到0~1
        if self.ot:
            # optimal transport 最优传输  attn_weight; [N,H',W'l,W'r] -> [N, H‘, W’l+1, W'r+1]
            attn_ot = self._optimal_transport(attn_weight, 10)  # 最优传输获得的attn_ot
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)

        # compute relative response (RR) at ground truth location 计算真值位置的相对响应
        if real_disp is not None:
            # find ground truth response (gt_response) and location (target) 发现真值响应（gt_response）和位置（目标）
            # 真值响应：attn中l和r的对应位置的特征[N, H‘, W’l] target左视图像素对应于右图中的像素位置索引
            output['gt_response'], target, output['gt_response_mask'] = self._compute_gt_location(scale, sampled_cols, sampled_rows, attn_ot[..., :-1, :-1], real_disp)
        else:
            output['gt_response'], output['gt_response_mask'] = None, None

        # regress low res disparity  回归低分辨率视差 [N, 1, W'l, W'r]
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # 计算未缩放的左右图对应位置偏移  N x H x W_left x W_right
        disp_pred_low_res, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1])  # 计算所得低分辨视差[N, H/4, W/4], norm_attn [N, H/4, W/4]
        # regress low res occlusion  回归低分辨率遮挡 [N, 1, H', W'l]
        occ_pred_low_res = self._compute_low_res_occ(matched_attn)

        # upsample and context adjust
        if sampled_cols is not None:
            output['disp_pred'], output['disp_pred_raw'], output['disp_pred_low_res'], output['occ_pred'] = self._upsample(x_l, disp_pred_low_res, occ_pred_low_res, scale)
            return [output['gt_response'], output['gt_response_mask'], output['disp_pred_low_res'], output['disp_pred_raw'], output['disp_pred']]
            # return [output['gt_response'], output['gt_response_mask'], output['disp_pred_low_res'], output['disp_pred']]
            # return [output['gt_response'], output['gt_response_mask'], output['disp_pred_raw'], output['disp_pred']]
            # return [output['gt_response'], output['gt_response_mask'], output['disp_pred']]
        else:
            output['disp_pred'] = disp_pred_low_res
            output['occ_pred'] = occ_pred_low_res
            return [output['gt_response'], output['disp_pred']]


def DispRegression(args):
    cal = build_context_adjustment_layer(args)

    if args.MODEL.SET.BM.DISPREGRESSION.REG_HEAD == 'ot':
        ot = True
    elif args.MODEL.SET.BM.DISPREGRESSION.REG_HEAD == 'softmax':
        ot = False
    else:
        raise Exception('Regression head type not recognized: ', args.regression_head)

    return RegressionHead(cal, ot)


if __name__ == '__main__':
    pass

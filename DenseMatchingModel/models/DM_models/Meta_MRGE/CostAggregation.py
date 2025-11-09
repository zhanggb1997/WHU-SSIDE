'''
Project    : RSDetec
FileName   : CostAttenSTTR .py
CreateTime : 2024/10/10 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''

from typing import Optional
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from models.DM_models.CSTR.misc import get_clones
from models.DM_models.OurNet4.MultiHeadAtten import MultiheadAttentionRelative


class CostAggregation(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        # self.MIM = MIM

        self.num_attn_layer = cfgs.MODEL.SET.BM.COSTFUSION.ATTEN_NUM
        hidden_dim = cfgs.MODEL.SET.BM.COSTFUSION.HIDDEN_DIM
        num_head = cfgs.MODEL.SET.BM.COSTFUSION.HEAD_NUM

        # 自注意力,复制num_attn_layers次
        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, num_head)
        self.self_attn_layers = get_clones(self_attn_layer, self.num_attn_layer)

        # 交叉注意力,复制num_attn_layers次
        cros_attn_layer = TransformerCrosAttnLayer(hidden_dim, num_head)
        self.cros_attn_layers = get_clones(cros_attn_layer, self.num_attn_layer)

    def _alternating_attn(self, feat: torch.Tensor, pos_enc: torch.Tensor, pos_indexes: Tensor, hn: int):
        """ 交替使用梯度检查点进行自我和交叉注意，以节省内存
        Alternate self and cross attention with gradient checkpointing to save memory

        :param feat: image feature concatenated from left and right, [W,2HN,C] (1/4)  从左到右串联的图像特征
        :param feat1: small image feature concatenated from left and right, [W,2HN,C] (1/8)  从左到右串联的小图像特征
        :param pos_enc: positional encoding, [2W/4-1, ChannelDim]  x轴位置编码
        :param pos_indexes: indexes to slice positional encoding, [W/4,W/4]  x轴切片位置编码的索引
        :param pos_enc1: positional encoding, [2W/8-1, ChannelDim]  小特征图x轴位置编码
        :param pos_indexes1: indexes to slice positional encoding, [W/8,W/8]  小特征图x轴切片位置编码的索引
        :param pos_enc_y: positional encoding along y axis, [2H/4-1, nChannelDim]  y轴位置编码
        :param pos_indexes_y: indexes to slice positional encoding along y axis, [H/4,H/4]  y轴索引到切片位置编码
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating, (自注意力 / 交叉注意力) = 交替使用，重复num_attn_layers=6/3次
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cros_attn_layers)):
            layer_idx = idx

            # checkpoint self attn, 自注意力的checkpoint, 用计算换取内存
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):  # 进行自注意力
                    return module(*inputs)

                return custom_self_attn

            # 进行自注意力
            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)

            # add a flag for last layer of cross attention, 判断是否是交叉注意力最后一层
            if idx == self.num_attn_layer - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn, 交叉注意力非最后一层
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            if idx == self.num_attn_layer - 1:
                feat, attn_weight, attn_weight_r = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:],
                                               pos_enc, pos_indexes)
                layer_idx = 0
                return attn_weight, attn_weight_r
            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:],
                                           pos_enc, pos_indexes)

        # layer_idx = 0
        # return attn_weight

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor,
                pos_enc: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N, C=128, H/4, W/4], 1/4左视图特征  [N, C, H', W']
        :param feat_right: feature descriptor of right image, [N, C=128, H/4, W/4], 1/4右视图特征  [N, C, H', W']
        :param pos_enc: relative positional encoding, (2xW/4-1, channel_dim=128) x轴位置编码  [2W'-1, channel_dim]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image , 返回交叉注意力值
        """

        bs, c, h, w = feat_left.shape

        # flatten NxCxHxW to WxHNxC , 维度转换
        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # [N, C, H', W'] -> [C, W', H', N] -> [C, W', H'N] -> [W', H'N, C]
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # [N, C, H', W'] -> [C, W', H', N] -> [C, W', H'N] -> [W', H'N, C]

        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding  移位rel-pos编码的索引
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)  # [W', 1]   [W'-1 W'-2 ... 1 0]
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)  # [1, W']   [0 1 ... W'-2 W'-1]
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # [W', W']  [W-1, W, W+1, ..., 2W-2, W-2, W-1, W, ..., 0, 1, 2, ..., W-1]
        else:
            pos_indexes = None

        # concatenate left and right features 合并左右视图特征
        feat = torch.cat([feat_left, feat_right], dim=1)  # [W', 2H'N, C]

        # compute attention 计算注意力
        attn_weight, attn_weight_r = self._alternating_attn(feat, pos_enc, pos_indexes, h*bs)  # [W', 2H'N, C]
        attn_weight = attn_weight.view(h, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image
        attn_weight_r = attn_weight_r.view(h, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image
        # [H/4N, W/4, W'/4] => [H/4, N, W/4, W'/4] => [N, H/4, W/4, W'/4]
        return attn_weight, attn_weight_r


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                ):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        # 特征标准化
        feat2 = self.norm(feat)  # [W', 2H'N, C]
        # 自注意力   QKV=feat2 [W', 2H'N, C], pos_x [2W'-1, C], pos_indexs_x [W'W']
        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos, pos_indexes=pos_indexes, is_self=True)

        feat = feat + feat2

        return feat


class TransformerCrosAttnLayer(nn.Module):
    """
    Cros attention layer
    """
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)
        self.norm1l = nn.LayerNorm(hidden_dim)
        self.norm1r = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, feat_left: Tensor, feat_right: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        """
        :param feat_left: left image feature, [W',H'N,C]
        :param feat_right: right image feature, [W',H'N,C]
        :param pos: pos encoding, [2W'-1,C]
        :param pos_indexes: indexes to slicer pos encoding [W',W']
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1l(feat_left)  # [W', 2H'N, C]
        feat_right_2 = self.norm1r(feat_right)  # [W', 2H'N, C]

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])  # [2W'-1, C]
        else:
            pos_flipped = pos
        # 交叉注意力 Q=featR2, KV=featL2   QKV=feat2 [W', 2H'N, C], pos [2W'-1, C], pos_indexs [W'W']
        feat_right_2, _, raw_attn_r = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped, pos_indexes=pos_indexes, is_self=False)

        feat_right = feat_right + feat_right_2

        # 视差有正负不使用atten mask
        # # update left features
        # # use attn mask for last layer 最后一层使用attn mask
        # if last_layer:
        #     w = feat_left_2.size(0)
        #     attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        # else:
        #
        attn_mask = None

        feat_right_2 = self.norm2(feat_right)  # [W', H'N, C]
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2, attn_mask=attn_mask, pos_enc=pos, pos_indexes=pos_indexes, is_self=False)

        feat_left = feat_left + feat_left_2  # [W', H'N, C]

        # concat features
        feat = torch.cat([feat_left, feat_right], dim=1)  # [W', H'N, C]

        if last_layer:
            return feat, raw_attn, raw_attn_r

        return feat, raw_attn  # [W', H'N, C],  [H'N, W', W']



    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular

        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask

if __name__ == '__main__':
    pass

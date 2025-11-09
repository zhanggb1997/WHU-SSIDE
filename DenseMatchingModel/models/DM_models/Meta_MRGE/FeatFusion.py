'''
Project    : RSDetec
FileName   : FeatFusionSwinT .py
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
from torchvision.models.densenet import _DenseBlock
import torch.nn.functional as F
from models.DM_models.CSTR.misc import center_crop


class ConvDown(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=2, padding=0, bias=True)

        elif scale == 4:
            self.convTrans = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels//2,
                    kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels//2),
                nn.Conv2d(
                    in_channels=out_channels//2, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=1, bias=True)
            )

    def forward(self, x: Tensor, skip: Tensor):
        out = self.convTrans(skip)
        out = torch.cat([out, x], 1)
        return out


class TransitionUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()
        if scale == 2:
            self.convTrans = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=2, padding=0, bias=True)

        elif scale == 4:
            self.convTrans = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ConvTranspose2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, stride=2, padding=0, bias=True)
            )

    def forward(self, x: Tensor, skip: Tensor):
        out_up = self.convTrans(x)
        out_up = center_crop(out_up, skip.size(2), skip.size(3))
        out = torch.cat([out_up, skip], 1)
        return out


class CatUp(nn.Module):
    """
    Scale the resolution up by transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 2):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: Tensor, skip: Tensor):
        out = self.conv(x)
        out = torch.cat([out, skip], 1)
        return out


class DoubleConv(nn.Module):
    """
    Two conv2d-bn-relu modules
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Tokenizer(nn.Module):
    """
    Expanding path of feature descriptor using DenseBlocks  使用DenseBlock扩展特征描述符的路径
    """

    def __init__(self, block_config: list, hidden_dim: int, backbone_feat_channel: list, growth_rate: int):
        super(Tokenizer, self).__init__()

        backbone_feat_channel.reverse()  # reverse so we have high-level first (lowest-spatial res) 反转，先对低分辨率结果进行处理
        block_config.reverse()

        self.block_config = block_config  # 块配置  [4, 4, 4, 4]
        self.growth_rate = growth_rate  # 生长率  4
        self.hidden_dim = hidden_dim  # 隐藏维度  128

        self.num_resolution = len(backbone_feat_channel)  # 多少级分辨率  3  [64， 128， 128]

        # 1/16 分辨率特征进行处理， 4, 128, 4, 16
        self.bottle_neck = _DenseBlock(block_config[0], backbone_feat_channel[0], 4, drop_rate=0.0, growth_rate=growth_rate)
        up = []
        dense_block = []
        prev_block_channels = growth_rate * block_config[0]


        # 1/8, 1/4, 1/1
        for i in range(self.num_resolution):
            if i == self.num_resolution - 1:
                up.append(ConvDown(1, 32, 4))
                dense_block.append(DoubleConv(prev_block_channels + 32, hidden_dim))
            else:
                up.append(TransitionUp(prev_block_channels, prev_block_channels))
                cur_channels_count = prev_block_channels + backbone_feat_channel[i + 1]
                dense_block.append(_DenseBlock(block_config[i + 1], cur_channels_count, 4, drop_rate=0.0, growth_rate=growth_rate))
                prev_block_channels = growth_rate * block_config[i + 1]


        self.up = nn.ModuleList(up)
        self.dense_block = nn.ModuleList(dense_block)

    def forward(self, features: list):
        """
        :param features:
            list containing feature descriptors at different spatial resolution
                0: [2N, C0, H//4, W//4]   64 x H/4 x W/4  ||
                1: [2N, C1, H//8, W//8]   128 x H/8 x W/8  ||
                2: [2N, C2, H//16, W//16]   128 x H/16 x W/16 ||
        :return: feature descriptor at full resolution [2N,C,H//8,W//8]_gwy
        """
        outputs = []

        features.reverse()  # [1/32, 1/16, 1/8, 1/4, 1/1]
        output = self.bottle_neck(features[0])  #
        output = output[:, -(self.block_config[0] * self.growth_rate):]  # take only the new features

        for i in range(self.num_resolution):   # (低分辨率特征 ↑采样) 和 (高分辨率特征) cat ;  然后进行dense卷积
            hs = self.up[i](output, features[i + 1])  # scale up and concat
            output = self.dense_block[i](hs)  # denseblock
            if i < self.num_resolution - 1:  # other than the last convolution block 非最后一个，则取最新的特征层
                output = output[:, -(self.block_config[i + 1] * self.growth_rate):]  # take only the new features
            else:
                # output = output
                output = output[:, -(self.block_config[i] * self.growth_rate):]
            outputs.append(output)

        # return output  # (2N x 128-64+64 x H/4 x W/4)
        return outputs  #


def FeatFusion(args):
    layer_channel = args.MODEL.SET.BM.FEATFUSION.LAYER_CHANNEL
    block_config = args.MODEL.SET.BM.FEATFUSION.BLOCK_CONFIG
    growth_rate = args.MODEL.SET.BM.FEATFUSION.GROWTH_RATE
    channel_dim = args.MODEL.SET.BM.FEATFUSION.CHANNEL_DIM
    return Tokenizer(block_config, channel_dim, layer_channel, growth_rate)


if __name__ == '__main__':
    pass

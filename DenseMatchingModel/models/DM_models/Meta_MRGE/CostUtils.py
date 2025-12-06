'''
Project    : RSDetec
FileName   : HMSMUtils .py
CreateTime : 2024/12/5 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import torch
from torch import nn
import torch.nn.functional as F

## 特征提取
def conv2d(inchannels, filters, kernel_size, strides, padding, dilation_rate):
    return nn.Conv2d(in_channels= inchannels, out_channels=filters, kernel_size=kernel_size,
                     stride=strides, padding=padding, dilation=dilation_rate)

def conv2d_bn(inchannels,filters, kernel_size, strides, padding, dilation_rate, activation):
    conv = nn.Conv2d(in_channels=inchannels, out_channels=filters, kernel_size=kernel_size,
                     stride=strides, padding=padding, dilation=dilation_rate, bias=False)
    bn = nn.BatchNorm2d(filters)
    relu = nn.ReLU()
    layers = [conv, bn]
    if activation:
        layers.append(relu)
    return nn.Sequential(*layers)



class CostConcatenation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(CostConcatenation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs): # N C, H ,W
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(F.pad(torch.cat((inputs[0][:, :, :,:i], inputs[1][:, :, :,-i:]), dim=1), (0, -i)))
            elif i > 0:
                cost_volume.append(F.pad(torch.cat((inputs[0][:, :, :,i:], inputs[1][:, :, :,:-i]), dim=1), (i, 0)))
            else:
                cost_volume.append(torch.cat((inputs[0], inputs[1]), dim=1))
        cost_volume = torch.stack(cost_volume, 2)
        return cost_volume #N C(D) H W



alpha = 0.2


def conv3d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

class Conv3dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, activation=True):
        super(Conv3dBn, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x

class TransConv3dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding, activation=True):
        super(TransConv3dBn, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding,
                                       bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x



class Hourglass(nn.Module):
    def __init__(self, filters):
        super(Hourglass, self).__init__()

        self.conv1 = Conv3dBn(filters, filters, 3, 1, 1, True)
        self.conv2 = Conv3dBn(filters, filters, 3, 1, 1, True)
        self.conv3 = Conv3dBn(filters, 2 * filters, 3, 2, 1, True)
        self.conv4 = Conv3dBn(2 * filters, 2 * filters, 3, 1, 1, True)
        self.conv5 = Conv3dBn(2 * filters, 2 * filters, 3, 2, 1, True)
        self.conv6 = Conv3dBn(2 * filters, 2 * filters, 3, 1, 1, True)
        self.conv7 = TransConv3dBn(2 * filters, 2 * filters, 4, 2, 1, True)
        self.conv8 = TransConv3dBn(2 * filters, filters, 4, 2, 1, True)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.conv2(x1)
        if x1.shape[2] % 2 != 0:
            x1 = nn.functional.pad(x1, (0, 1, 0, 1, 0, 0))
        x2 = self.conv3(x1)
        x2 = self.conv4(x2)
        if x2.shape[2] % 2 != 0:
            x2 = nn.functional.pad(x2, (0, 1, 0, 1, 0, 0))
        x3 = self.conv5(x2)
        x3 = self.conv6(x3)
        x4 = self.conv7(x3)
        x4 += x2
        x5 = self.conv8(x4)
        if x1.shape[2] != x5.shape[2]:
            x1 = nn.functional.pad(x1, (0, 0, 1, 1, 1, 1))
        x5 += x1

        return x5  # [N, C,D,H ,W] # differen for tensorflow

class FeatureFusion(nn.Module):
    def __init__(self, infeatures, units):
        super(FeatureFusion, self).__init__()

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')  # 使用Upsample实现上采样
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))  # 在PyTorch中，全局平均池化层使用AdaptiveAvgPool3d
        self.fc1 = nn.Linear(infeatures, units)  # 在PyTorch中，全连接层使用Linear
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(units, units)  # 再次使用全连接层
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, inputs):
        x1 = self.upsample(inputs[0])
        x2 = torch.add(x1, inputs[1])
        v = self.avg_pool3d(x2)[:, :, 0, 0, 0]
        v = self.fc1(v)
        v = F.relu(v)
        v = self.fc2(v)
        v = self.sigmoid(v)
        v1 = 1.0 - v
        v = v.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        v1 = v1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        x3 = torch.mul(x1, v)
        x4 = torch.mul(inputs[1], v1)
        x = torch.add(x3, x4)

        return x


class FeatureFusionX(nn.Module):
    def __init__(self, infeatures, units, min_D, max_D, up_n=2):
        super(FeatureFusionX, self).__init__()

        self.min_D = min_D
        self.max_D = max_D

        self.upsample = nn.Upsample(scale_factor=(up_n, 1, 1), mode='nearest')  # 使用Upsample实现上采样
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))  # 在PyTorch中，全局平均池化层使用AdaptiveAvgPool3d
        self.fc1 = nn.Linear(infeatures, units)  # 在PyTorch中，全连接层使用Linear
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(units, units)  # 再次使用全连接层
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, inputs):
        x1 = self.upsample(inputs[0][:,:,self.min_D:self.max_D])
        x2 = torch.add(x1, inputs[1])
        v = self.avg_pool3d(x2)[:, :, 0, 0, 0]
        v = self.fc1(v)
        v = F.relu(v)
        v = self.fc2(v)
        v = self.sigmoid(v)
        v1 = 1.0 - v
        v = v.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        v1 = v1.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x2)
        x3 = torch.mul(x1, v)
        x4 = torch.mul(inputs[1], v1)
        x = torch.add(x3, x4)

        return x


## 视差计算
class Estimation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0, interval=4.0, input_channels=32):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.interval = int(interval)
        self.conv = nn.Conv3d(input_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        x = self.conv(inputs)     # [N, 1, D, H, W]
        x = x.squeeze(1)  # [N, D, H, W]
        assert x.shape[1] == (self.max_disp - self.min_disp) // self.interval
        # candidates = torch.linspace(float(self.min_disp) // self.interval, float(self.max_disp - 1) // self.interval, int(self.max_disp - self.min_disp) // self.interval).cuda()
        candidates = torch.arange(self.min_disp, self.max_disp, self.interval, dtype=x.dtype, device=x.device)
        probabilities = F.softmax(x, dim=1)
        disparities = torch.sum(candidates.view(1, (self.max_disp - self.min_disp) // self.interval, 1, 1) * probabilities, dim=1, keepdim=True)
        return disparities



## 时差精细化
class conv_bn_act(nn.Module):
    def __init__(self, inchannels, filters, kernel_size, strides, padding, dilation_rate):
        super(conv_bn_act, self).__init__()
        self.conv = nn.Conv2d(inchannels, filters, kernel_size=kernel_size, stride=strides, padding=padding, dilation=dilation_rate, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class Refinement(nn.Module):
    def __init__(self, filters):
        super(Refinement, self).__init__()
        self.conv0 = nn.Sequential(
            conv_bn_act(3, 32, 3, 1, 1, 1),
            conv_bn_act(32, 1, 3, 1, 1, 1),
        )
        self.conv1 = conv_bn_act(4, filters, 3, 1, 1, 1)
        self.conv2 = conv_bn_act(filters, filters, 3, 1, 1,  1)
        self.conv3 = conv_bn_act(filters, filters, 3, 1, 2, 2)
        self.conv4 = conv_bn_act(filters, filters, 3, 1, 3, 3)
        self.conv5 = conv_bn_act(filters, filters, 3, 1, 1, 1)
        self.conv6 = conv2d(filters, 1, 3, 1, 1, 1)

    def forward(self, inputs):
        # inputs: [disparity, rgb, gx, gy]
        # assert len(inputs) == 4

        scale_factor = inputs[1].shape[2] / inputs[0].shape[2]
        disp = F.interpolate(inputs[0], size=(inputs[1].shape[2], inputs[1].shape[3]), mode='bilinear', align_corners=True)
        disp = disp * scale_factor
        disp2x = F.interpolate(inputs[-2], size=(inputs[1].shape[2], inputs[1].shape[3]), mode='bilinear', align_corners=True)
        disp2x = disp2x * 2
        disp4x = F.interpolate(inputs[-1], size=(inputs[1].shape[2], inputs[1].shape[3]), mode='bilinear', align_corners=True)
        disp4x = disp4x * 1

        disp_ = self.conv0(torch.cat((disp, disp2x, disp4x), dim=1))

        concat = torch.cat((disp_, inputs[1], inputs[2], inputs[3]), dim=1)
        # concat = torch.cat((disp, inputs[1], inputs[2], inputs[3]), dim=1)
        delta = self.conv1(concat)
        delta = self.conv2(delta)
        delta = self.conv3(delta)
        delta = self.conv4(delta)
        delta = self.conv5(delta)
        delta = self.conv6(delta)
        disp_final = disp + delta

        return disp_final




class GWCCostConcat():

    def __init__(self, min_disp, max_disp, num_groups):
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_groups = num_groups

    def __call__(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, self.num_groups, int(self.max_disp - self.min_disp), H, W])
        for i in range(self.min_disp, self.max_disp):
            if i > 0:
                volume[:, :, i, :, i:] = self.groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], self.num_groups)
            elif i == 0:
                volume[:, :, i, :, :] = self.groupwise_correlation(refimg_fea, targetimg_fea, self.num_groups)
            else:
                volume[:, :, i, :, :i] = self. groupwise_correlation(refimg_fea[:, :, :, :i], targetimg_fea[:, :, :, -i:], self.num_groups)
        volume = volume.contiguous()
        return volume

        # cost_volume = []
        # for i in range(self.min_disp, self.max_disp):
        #     if i < 0:
        #         cost_volume.append(F.pad(torch.cat((refimg_fea[:, :, :,:i], targetimg_fea[:, :, :,-i:]), dim=1), (0, -i)))
        #     elif i > 0:
        #         cost_volume.append(F.pad(torch.cat((refimg_fea[:, :, :,i:], targetimg_fea[:, :, :,:-i]), dim=1), (i, 0)))
        #     else:
        #         cost_volume.append(torch.cat((refimg_fea, targetimg_fea), dim=1))
        #
        # cost_volume = torch.stack(cost_volume, 2)
        # return cost_volume



    def groupwise_correlation(self, fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return cost



class DispRegress(nn.Module):
    def __init__(self, mindisp, maxdisp, input_channels):
        super(DispRegress, self).__init__()
        self.classifier = nn.Conv3d(input_channels, 1, 3, 1, 1, bias=False)
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, feat):
        cost_volume = self.classifier(feat)
        prob_volume = F.softmax(cost_volume.squeeze(1), dim=1)
        assert len(prob_volume.shape) == 4
        disp_values = torch.arange(self.mindisp, self.maxdisp, 1, dtype=feat.dtype, device=feat.device)
        disp_values = disp_values.view(1, (self.maxdisp-self.mindisp), 1, 1)
        return torch.sum(prob_volume * disp_values, 1, keepdim=True)

if __name__ == '__main__':
    pass

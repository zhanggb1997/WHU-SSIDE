'''
Project    : RSDetec
FileName   : utils .py
CreateTime : 2024/10/9 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from itertools import repeat
import collections.abc
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join(''))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or (head_dim ** -0.5)
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias  定义相对位置偏差参数表
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window  获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # (B*win_num, win_size*win_size, C) -> (B*win_num, 3*win_size*win_size, C) -> (B*win_num, win_size*win_size, 3, H, C/H) -> (3, B*win_num, H, win_size*win_size, C/H)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #  q, k, v (B*win_num, H, win_size*win_size, C/H)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B*win_num, H, win_size*win_size, win_size*win_size)
        #
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # (B*win_num, H, win_size*win_size, C/H) -> (B*win_num, win_size*win_size, C)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # (B*win_num, win_size*win_size, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  mlp隐藏dim与嵌入dim的比率
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True  如果为True，则向查询、键、值添加可学习的偏差
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.  覆盖head\u dim**-0.5（如果设置）的默认qk刻度
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # (B, H * W, C) -> (B, H, W, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows 使用窗口切分 (B, H, W, C) -> (B*win_num, win_size, win_size, C)
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows 使用窗口切分 (B, H, W, C) -> (B*win_num, win_size, win_size, C)
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # partition windows 使用窗口切分 (B*win_num, win_size, win_size, C) -> (B*win_num, win_size * win_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA 窗口-多头自注意力/偏移窗口-多头自注意力  (B*win_num, win_size*win_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows (B*win_num, win_size, win_size, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift  (B*win_num, win_size, win_size, C) -> (B, H, W, C)
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops



class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample([in_res * 2 for in_res in input_resolution], dim=int(dim/2), norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=1024, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # (B, C, H, W) -> (B, C, H/ps, W/ps) -> (B, C, H/ps * W/ps) -> (B, H/ps * W/ps, C)  C=embed_dim
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)  # (B, H/ps * W/ps, C)
        return x


class FeatExtract(nn.Module):
    def __init__(self, cfgs, MIM=False):
        super().__init__()
        self.cfgs = cfgs

        self.MIM = MIM

        # 配置信息检索
        img_size = self.cfgs.TRAIN.IMAGE_SIZE  # 1024
        img_chan = self.cfgs.MODEL.NUM_INCHANNELS
        patch_size = self.cfgs.MODEL.SET.BM.FEATEXTRACT.PATCH_SIZE  # 4
        window_size = self.cfgs.MODEL.SET.BM.FEATEXTRACT.WINDOWS_SIZE  # 7
        embed_dim = self.cfgs.MODEL.SET.BM.FEATEXTRACT.EMBED_DIM # 96
        depths = self.cfgs.MODEL.SET.BM.FEATEXTRACT.DEPTHS  # [2, 2, 6, 2]
        num_heads = self.cfgs.MODEL.SET.BM.FEATEXTRACT.HAEDS_NUM  # [3, 6, 12, 24],
        num_layer = len(depths)
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None,
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        patch_norm = True,
        use_checkpoint = False
        fused_window_process = False
        self.ape = False
        self.num_features = int(embed_dim * 2 ** (num_layer - 1))

        # 模型部分
        normlayer = nn.LayerNorm
        # 嵌入头
        if self.cfgs.DATA.ADD_META:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=img_chan, embed_dim=embed_dim-1, norm_layer=normlayer)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=img_chan, embed_dim=embed_dim, norm_layer=normlayer)
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=img_chan, embed_dim=embed_dim, norm_layer=normlayer)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        # # absolute position embedding  绝对位置嵌入
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth  随机深度 drop path?
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # swinT迭代
        self.layers = nn.ModuleList()
        for layer_i in range(num_layer):
            layer = BasicLayer(
                # dim=int(embed_dim + 2) if layer_i==0 and self.cfgs.DATA.ADD_META else int(embed_dim * 2 ** layer_i),
                dim=int((embed_dim) * (2 ** layer_i)),
                input_resolution=(patches_resolution[0] // (2 ** layer_i), patches_resolution[1] // (2 ** layer_i)),
                depth=depths[layer_i],
                num_heads=num_heads[layer_i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:layer_i]):sum(depths[:layer_i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (layer_i > 0) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process
            )
            self.layers.append(layer)

            self.norm = norm_layer(self.num_features)

        # MIM mask
        if self.MIM:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.mask_token, mean=0., std=.02)

        # META Encoder
        if self.cfgs.DATA.ADD_META:
            self.meta_token = ModalEncoder(cfgs)


    def forward(self, x, mask=None, metas=None, meta_view_mode=None):
        x_raw = x
        # feature 特征提取
        # (B, C, H, W) -> (B, H / ps * W / ps, C)  C=embed_dim
        x = self.patch_embed(x)
        B, H_W_, C = x.shape

        if self.cfgs.DATA.ADD_META:
            meta = self.meta_token(metas, meta_view_mode)
            x = torch.cat((x, meta), 2)
            C = C + 1
            # meta_tokens = self.meta_token(metas, meta_view_mode)

            # gamma = meta_tokens[0][:, :, :self.cfgs.MODEL.SET.BM.FEATEXTRACT.EMBED_DIM]
            # beta  = meta_tokens[0][:, :, self.cfgs.MODEL.SET.BM.FEATEXTRACT.EMBED_DIM:]
            #
            # x = x * gamma + beta
            #
            # # 遍历层提取特征
            # x_layers = [x_raw]
            # for layer_no, layer in enumerate(self.layers):
            #     x = layer(x)
            #     if layer_no == len(self.layers) - 1:
            #         x = self.norm(x)
            #
            #     gamma = meta_tokens[layer_no+1][:, :, :self.cfgs.MODEL.SET.BM.FEATFUSION.LAYER_CHANNEL[3-layer_no]]
            #     beta  = meta_tokens[layer_no+1][:, :, self.cfgs.MODEL.SET.BM.FEATFUSION.LAYER_CHANNEL[3-layer_no]:]
            #
            #     x = x * gamma + beta
            #
            #     x_ = x.view(B, int((H_W_)**0.5/(2**layer_no)), int((H_W_)**0.5/(2**layer_no)), int(C*(2**layer_no))).permute(0, 3, 1, 2)
            #
            #     x_layers.append(x_ )
            #
            # return x_layers


        # MIM mask
        if self.MIM:
            assert mask is not None
            mask_tokens = self.mask_token.expand(B, H_W_, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + (mask_tokens * w)

        # absolute position embedding  绝对位置嵌入
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 遍历层提取特征
        x_layers = [x_raw]
        for layer_no, layer in enumerate(self.layers):
            x = layer(x)
            if layer_no == len(self.layers) - 1:
                x = self.norm(x)
            x_ = x.view(B, int((H_W_)**0.5/(2**layer_no)), int((H_W_)**0.5/(2**layer_no)), int(C*(2**layer_no))).permute(0, 3, 1, 2)
            x_layers.append(x_)

        return x_layers


class ModalEncoder1(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.embed_dim = self.cfgs.MODEL.SET.BM.FEATEXTRACT.EMBED_DIM
        self.layer_dim = self.cfgs.MODEL.SET.BM.FEATFUSION.LAYER_CHANNEL

        self.layer_mlp_sa = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 32),
        )

        self.layer_mlp_md = nn.Sequential(
            nn.Linear(1, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 32),
        )

        self.layer_mlp_za = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 32),
        )

        # self.layer_mlp_ll = nn.Sequential(
        #     nn.Linear(2, 24),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(24, 48),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(48, 32),
        # )

        self.layer_mlp_ll = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 160),
        )

        self.layer_mlp_sza = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 32),
        )

        # self.layer_mlp5 = nn.Sequential(
        #     nn.Linear(2, 48),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(48, 96),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(96, 128),
        # )

        #
        self.final_proj0 = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 96),
        )

        self.final_proj1 = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 96),
        )

        self.final_proj2 = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 192),
        )

        self.final_proj3 = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 384),
        )

        self.final_proj4 = nn.Sequential(
            nn.Linear(160, 160),
            nn.ReLU(inplace=True),
            nn.Linear(160, 768),
        )


    def forward(self, metas, meta_view_mode):
        """
        meta_infos: 包含 [分辨率, 时间] 等连续属性
        """

        # meta_raw_sm = torch.cat((metas['stereo_angle'].unsqueeze(0),
        #                       metas['month_diff'].unsqueeze(0),
        #                       ), 1)
        #
        meta_raw_ll = torch.cat((
                              metas[meta_view_mode + '_lat'].unsqueeze(0),
                              metas[meta_view_mode + '_lon'].unsqueeze(0)
                              ), 1)

        # meta_raw_za = torch.cat((
        #                       metas[meta_view_mode + '_zenith'].unsqueeze(0),
        #                       metas[meta_view_mode + '_azimuth'].unsqueeze(0)
        #                       ), 1)
        #
        #
        # meta_raw_sza = torch.cat((
        #                       metas[meta_view_mode + '_sunzenith'].unsqueeze(0),
        #                       metas[meta_view_mode + '_sunazimuth'].unsqueeze(0)
        #                       ), 1)
        #
        # meta_raw_sa = metas['stereo_angle'].unsqueeze(0)
        # meta_raw_md = metas['month_diff'].unsqueeze(0)


        meta_mlp_ll = self.layer_mlp_ll(meta_raw_ll)
        # meta_mlp_sa = self.layer_mlp_sa(meta_raw_sa)
        # meta_mlp_md = self.layer_mlp_md(meta_raw_md)
        # meta_mlp_za = self.layer_mlp_za(meta_raw_za)
        # meta_mlp_sza = self.layer_mlp_sza(meta_raw_sza)

        # # # 拼接 (B, embed_dim)
        # meta_cat = torch.cat([meta_mlp_sm, meta_mlp_za, meta_mlp_sza, meta_mlp_ll], dim=-1)
        # meta_cat = torch.cat([meta_mlp_sa, meta_mlp_md, meta_mlp_za, meta_mlp_sza, meta_mlp_ll], dim=-1)
        meta_cat = meta_mlp_ll
        # meta_cat = torch.cat([meta_mlp_sm], dim=-1)
        # #
        # # # 再映射到 (B, embed_dim)
        meta_vec0 = self.final_proj0(meta_cat).unsqueeze(2)
        meta_vec1 = self.final_proj1(meta_cat).unsqueeze(2)
        meta_vec2 = self.final_proj2(meta_cat).unsqueeze(2)
        meta_vec3 = self.final_proj3(meta_cat).unsqueeze(2)
        meta_vec4 = self.final_proj4(meta_cat).unsqueeze(2)
        # meta_token = meta_vec.repeat(1, 1, 256).view(1, 256 * 256, 1)
        #
        # # 增加一个 Token 维度: (B, 1, embed_dim)
        B, _, _ = meta_vec1.shape
        meta_token0 = meta_vec0.view(B, 1, self.embed_dim*2)
        meta_token1 = meta_vec1.view(B, 1, self.layer_dim[3]*2)
        meta_token2 = meta_vec2.view(B, 1, self.layer_dim[2]*2)
        meta_token3 = meta_vec3.view(B, 1, self.layer_dim[1]*2)
        meta_token4 = meta_vec4.view(B, 1, self.layer_dim[0]*2)

        return [meta_token0, meta_token1, meta_token2, meta_token3, meta_token4]


class ModalEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.embed_dim = self.cfgs.MODEL.SET.BM.FEATEXTRACT.EMBED_DIM

        # self.layer_mlp_sa = nn.Sequential(
        #     nn.Linear(1, 24),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(24, 48),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(48, 32),
        # )
        #
        # self.layer_mlp_md = nn.Sequential(
        #     nn.Linear(1, 24),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(24, 48),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(48, 32),
        # )
        #
        # self.final_proj = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 256),
        # )


        # US3D
        self.layer_mlp5 = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 128),
        )
        # self.layer_mlp_samd = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        #
        # self.layer_mlp_za = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        #
        # self.layer_mlp_ll = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        #
        # self.layer_mlp_sza = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )

        self.final_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )


        # # Our WHU-GF7-SSIDE
        #
        # self.layer_mlp_za = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        # self.layer_mlp_ll = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        # self.layer_mlp_sza = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 32),
        # )
        #
        #
        #
        # self.final_proj = nn.Sequential(
        #     nn.Linear(96, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 256),
        # )


        # Our WHU-GF7-SSIDE

        # self.layer_mlp_ll = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 96),
        # )
        #
        # self.final_proj = nn.Sequential(
        #     nn.Linear(96, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 256),
        # )

    def forward(self, metas, meta_view_mode):
        """
        meta_infos: 包含 [分辨率, 时间] 等连续属性
        """
        # # Our WHU-GF7-SSIDE
        # meta_raw_ll = torch.cat((
        #                       metas[meta_view_mode + '_lat'].unsqueeze(0),
        #                       metas[meta_view_mode + '_lon'].unsqueeze(0)
        #                       ), 1)
        # #
        # meta_raw_za = torch.cat((
        #                       metas[meta_view_mode + '_zenith'].unsqueeze(0),
        #                       metas[meta_view_mode + '_azimuth'].unsqueeze(0)
        #                       ), 1)
        #
        # meta_raw_sza = torch.cat((
        #                       metas[meta_view_mode + '_sunzenith'].unsqueeze(0),
        #                       metas[meta_view_mode + '_sunazimuth'].unsqueeze(0)
        #                       ), 1)

        # US3D
        # meta_raw_sa = metas['stereo_angle'].unsqueeze(0)
        # meta_raw_md = metas['month_diff'].unsqueeze(0)

        meta_raw_samd = torch.cat((
                              metas['stereo_angle'].unsqueeze(0),
                              metas['month_diff'].unsqueeze(0)
                              ), 1)


        # meta_mlp_ll = self.layer_mlp_ll(meta_raw_ll)
        # meta_mlp_za = self.layer_mlp_za(meta_raw_za)
        # meta_mlp_sza = self.layer_mlp_sza(meta_raw_sza)
        # meta_mlp_samd = self.layer_mlp_samd(meta_raw_samd)
        # meta_mlp_sa = self.layer_mlp_sa(meta_raw_sa)
        # meta_mlp_md = self.layer_mlp_md(meta_raw_md)
        meta_mlp_samd = self.layer_mlp5(meta_raw_samd)
        meta_vec = self.final_proj(meta_mlp_samd).unsqueeze(2)
        meta_token = meta_vec.repeat(1, 1, 256).view(1, 256 * 256, 1)


        # # # 拼接 (B, embed_dim)
        # meta_cat = torch.cat([meta_mlp_za, meta_mlp_sza, meta_mlp_ll], dim=-1)
        # meta_cat = torch.cat([meta_mlp_sa, meta_mlp_md], dim=-1)
        # meta_cat = torch.cat([meta_mlp_sa, meta_mlp_md, meta_mlp_za, meta_mlp_sza, meta_mlp_ll], dim=-1)
        # meta_cat = torch.cat([meta_mlp_samd, meta_mlp_za, meta_mlp_sza, meta_mlp_ll], dim=-1)
        # meta_cat = torch.cat([meta_mlp_ll], dim=-1)
        # meta_cat = torch.cat([meta_mlp_sza], dim=-1)
        # meta_cat = torch.cat([meta_mlp_sm], dim=-1)
        # #
        # # # # 再映射到 (B, embed_dim)
        # meta_vec = self.final_proj(meta_cat).unsqueeze(2)
        # meta_token = meta_vec.repeat(1, 1, 256).view(1, 256 * 256, 1)
        #
        # # # 增加一个 Token 维度: (B, 1, embed_dim)
        # B, _, _ = meta_vec.shape
        # meta_token = meta_vec.view(B, 1, self.embed_dim)

        return meta_token


if __name__ == '__main__':
    pass

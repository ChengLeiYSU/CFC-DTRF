import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as models
import math
from torch.nn import Conv2d
from torch.nn import LeakyReLU
import numpy as np
import time
import torch as th
import functools
from torch import einsum
from pdb import set_trace as stx
import timeit
import numbers
from torch.nn import ModuleList

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


vgg = models.vgg16(pretrained=True)

for param in vgg.parameters():
    param.requires_grad = False


color_layer = vgg.features[:4]
content_layer = vgg.features[:10]

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3,
                      stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel,
                                kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class UNet(nn.Module):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()

        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(
            dim*2, dim*2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(
            dim*4, dim*4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(
            dim*8, dim*8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x


class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):

        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))/np.sqrt(int(c/self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*3)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x

class CATransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CATransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.reshape((B, C, H, W))

        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        x = x.permute(0, 2, 3, 1)
        B_0, H_0, W_0, C_0 = x.shape
        L_0 = H_0 * W_0
        x = x.reshape(B_0, L_0, C_0)

        return x


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C //
                                 self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads,
                                         C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(
            B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(
            B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d, k_e), dim=2)
        v = torch.cat((v_d, v_e), dim=2)
        return q, k, v

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.heads = num_heads

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None, ):
        # print("shiyong")      #四维张量
        x = x.permute(0, 3, 1, 2)
        h = self.heads
        b, c, l, w = x.shape
        X = rearrange(x, 'b c l w -> b (l w) c', b=b, c=c, l=l, w=w)
        B_, N, C = X.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(
            relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

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


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.project_in = nn.Conv2d(dim, hidden_dim*2, kernel_size=1)

        self.dwconv = nn.Conv2d(
            hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, groups=hidden_dim*2)

        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        return x

def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=4, stride=2, padding=1),

        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel,
                               kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class InputProj(nn.Module):
    def __init__(self, in_channel=32, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SWCATransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff', se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, se_layer=se_layer)

        # self.attn_1 = ColorAttention(dim, num_heads, bias=False)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        # print(x.shape)
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        if self.shift_size > 0:  # 滑动窗口算法   滑动窗口注意力机制计算注意力掩码，以区分不同窗口内的元素是否属于同一区域
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1)).type_as(x).detach()  # 1 H W 1
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.win_size)  # nW, win_size, win_size, 1
            mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.type_as(x)
        else:
            attn_mask = None

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.win_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(
            attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        # print(x.shape)
        return x

class BasicSWCATransformerBlockformerLayer_content(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', se_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            SWCATransformerBlock(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0 if (
                                      i % 2 == 0) else win_size // 2,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop=drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(
                                      drop_path, list) else drop_path,
                                  norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
                # print(x.shape)

            else:
                x = blk(x)

        return x

class SWCAformer_content(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]
        self.input_proj = InputProj(
            in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(
            in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim,
                                                        output_dim=embed_dim,
                                                        input_resolution=(img_size,
                                                                          img_size),
                                                        depth=depths[0],
                                                        num_heads=num_heads[0],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=enc_dpr[sum(
                                                            depths[:0]):sum(depths[:1])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)
        self.encoderlayer_1 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*2,
                                                        output_dim=embed_dim*2,
                                                        input_resolution=(img_size // 2,
                                                                          img_size // 2),
                                                        depth=depths[1],
                                                        num_heads=num_heads[1],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=enc_dpr[sum(
                                                            depths[:1]):sum(depths[:2])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)
        self.encoderlayer_2 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*4,
                                                        output_dim=embed_dim*4,
                                                        input_resolution=(img_size // (2 ** 2),
                                                                          img_size // (2 ** 2)),
                                                        depth=depths[2],
                                                        num_heads=num_heads[2],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=enc_dpr[sum(
                                                            depths[:2]):sum(depths[:3])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)
        self.encoderlayer_3 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*8,
                                                        output_dim=embed_dim*8,
                                                        input_resolution=(img_size // (2 ** 3),
                                                                          img_size // (2 ** 3)),
                                                        depth=depths[3],
                                                        num_heads=num_heads[3],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=enc_dpr[sum(
                                                            depths[:3]):sum(depths[:4])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*16,
                                              output_dim=embed_dim*16,
                                              input_resolution=(img_size // (2 ** 4),
                                                                img_size // (2 ** 4)),
                                              depth=depths[4],
                                              num_heads=num_heads[4],
                                              win_size=win_size,
                                              mlp_ratio=self.mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              drop_path=conv_dpr,
                                              norm_layer=norm_layer,
                                              use_checkpoint=use_checkpoint,
                                              token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)

        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*16,
                                                        output_dim=embed_dim*16,
                                                        input_resolution=(img_size // (2 ** 3),
                                                                          img_size // (2 ** 3)),
                                                        depth=depths[5],
                                                        num_heads=num_heads[5],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=dec_dpr[:depths[5]],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*8,
                                                        output_dim=embed_dim*8,
                                                        input_resolution=(img_size // (2 ** 2),
                                                                          img_size // (2 ** 2)),
                                                        depth=depths[6],
                                                        num_heads=num_heads[6],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=dec_dpr[sum(
                                                            depths[5:6]):sum(depths[5:7])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*4,
                                                        output_dim=embed_dim*4,
                                                        input_resolution=(img_size // 2,
                                                                          img_size // 2),
                                                        depth=depths[7],
                                                        num_heads=num_heads[7],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=dec_dpr[sum(
                                                            depths[5:7]):sum(depths[5:8])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicSWCATransformerBlockformerLayer_content(dim=embed_dim*2,
                                                        output_dim=embed_dim*2,
                                                        input_resolution=(img_size,
                                                                          img_size),
                                                        depth=depths[8],
                                                        num_heads=num_heads[8],
                                                        win_size=win_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                                        drop_path=dec_dpr[sum(
                                                            depths[5:8]):sum(depths[5:9])],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint,
                                                        token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x):
        # print(x.shape)
        # Input Projection
        y = self.input_proj(x)
        # print(y.shape)
        y = self.pos_drop(y)
        # Encoder
        # print(y.shape)      #torch.Size([1, 65536, 32])
        conv0 = self.encoderlayer_0(y)
        # print(conv0.shape)
        pool0 = self.dowsample_0(conv0)
        # print(pool0.shape)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y


class color_content(nn.Module):
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(color_content, self).__init__()

        self.Up = up_conv(64, 64)
        self.mode_color = MSCATransformer(in_channels=64,     
                                          dim=64,              
                                          num_blocks=6,        
                                          num_heads=1,         
                                          scales=[1, 2, 4],    
                                          num_experts=4).to(device)
        self.model_1 = SWCAformer_content(in_chans=128)

    def forward(self, input_0, input_1):

        in_0 = input_0
        in_1 = input_1

        out_0 = self.mode_color(in_0)
        # print(out_0.shape)
        out_1 = self.model_1(in_1)

        return out_0, out_1


class catimage(nn.Module):
    def __init__(self):
        super(catimage, self).__init__()

        self.conv_transpose_1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128,
            kernel_size=4, stride=2, padding=1
        )
        self.conv_transpose_0 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64,
            kernel_size=4, stride=2, padding=1
        )
        
        self.net_0 = ResnetGenerator(
            128, 64, 64, norm_layer=nn.BatchNorm2d, 
            use_dropout=False, n_blocks=3, use_residual_features=True)
        self.net_1 = ResnetGenerator(
            64, 3, 64, norm_layer=nn.BatchNorm2d, 
            use_dropout=False, n_blocks=6, use_residual_features=True)

    def forward(self, image_0, image_1, res_color=None, res_content=None):
        image_1 = self.conv_transpose_1(image_1)
        image_0 = self.conv_transpose_0(image_0)
        

        image_1 = self.net_0(image_1, residual_content=res_content)

        img = image_1 + image_0
        
        img = self.net_1(img, residual_color=res_color)

        return img



class zongtimox(nn.Module):
    def __init__(self):
        super(zongtimox, self).__init__()

        self.colorconten = color_content()
        self.shengcheng = catimage()  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = models.vgg16(pretrained=True)
        self.color_layer = vgg.features[:4].to(device)
        self.content_layer = vgg.features[:9].to(device)
        self.convhalf = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

    def forward(self, rawimage):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        raw_color_features = self.color_layer(rawimage)
        raw_color_features = self.convhalf(raw_color_features)
        raw_content_features = self.content_layer(rawimage)


        jieshou_color, jieshou_content = self.colorconten(
            raw_color_features, raw_content_features)


        img = self.shengcheng(jieshou_color, jieshou_content, 
                             res_color=jieshou_color, 
                             res_content=jieshou_content)
        
        jieshou_color = self.conv_transpose(jieshou_color)
        return jieshou_color, jieshou_content, img



class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 use_residual_features=False):  # 新增参数
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.use_residual_features = use_residual_features

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.encoder = nn.Sequential(*model)
        
        # ResNet blocks
        mult = 2 ** n_downsampling
        self.res_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.res_blocks.append(
                ResnetBlock(ngf * mult, padding_type=padding_type,
                           norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            )
        
        
        if self.use_residual_features:
            self.fusion_color = nn.Conv2d(64 + ngf * mult, ngf * mult, kernel_size=1, bias=use_bias)
            self.fusion_content = nn.Conv2d(128 + ngf * mult, ngf * mult, kernel_size=1, bias=use_bias)
            self.fusion_norm = norm_layer(ngf * mult)
        
        # Decoder
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input, residual_color=None, residual_content=None):
        x = self.encoder(input)
        
        # 通过ResNet blocks
        fusion_point_1 = len(self.res_blocks) // 3
        fusion_point_2 = 2 * len(self.res_blocks) // 3
        
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            
           
            if i == fusion_point_1 and self.use_residual_features and residual_color is not None:
                B, C, H, W = x.shape
                
                res_color = F.interpolate(residual_color, size=(H, W), 
                                         mode='bilinear', align_corners=False)
                
                x_fused = torch.cat([x, res_color], dim=1)
                x = x + self.fusion_norm(self.fusion_color(x_fused))
            
            
            if i == fusion_point_2 and self.use_residual_features and residual_content is not None:
                B, C, H, W = x.shape
                
                res_content = F.interpolate(residual_content, size=(H, W), 
                                           mode='bilinear', align_corners=False)
                
                x_fused = torch.cat([x, res_content], dim=1)
                x = x + self.fusion_norm(self.fusion_content(x_fused))
        
        x = self.decoder(x)
        return x



class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p,
                                 bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class _equalized_conv2d(th.nn.Module):
    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """
        from torch.nn.modules.utils import _pair
        from numpy import sqrt, prod

        super().__init__()

        # define the weight and bias if to be used
        self.weight = th.nn.Parameter(th.nn.init.normal_(
            th.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))

        fan_in = prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = sqrt(2) / sqrt(fan_in)

    def forward(self, x):
        from torch.nn.functional import conv2d

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class PixelwiseNorm(th.nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch, use_eql=True):
        super(up_conv, self).__init__()
        self.conv_1 = _equalized_conv2d(in_ch, out_ch, (1, 1),
                                        pad=0, bias=True)
        self.conv_2 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                        pad=1, bias=True)
        self.conv_3 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                        pad=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        from torch.nn.functional import interpolate

        x = interpolate(x, scale_factor=2, mode="bilinear")
        y = self.conv_1(self.lrelu(self.pixNorm(x)))
        residual = y
        y = self.conv_2(self.lrelu(self.pixNorm(y)))
        y = self.conv_3(self.lrelu(self.pixNorm(y)))
        y = y+residual

        return y


class to_rgb(nn.Module):

    def __init__(self, inchannels, use_eql=True):
        super(to_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(224, 3, (1, 1), bias=True)
        else:
            self.conv_1 = nn.Conv2d(96, 3, (1, 1), bias=True)

    def forward(self, x):

        y = self.conv_1(x)

        return y


class AdaptiveNorm(nn.Module):

    def __init__(self, dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim)
        # 动态参数估计网络
        reduction = 4
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim // reduction, dim * 2, 1)

    def forward(self, x):
        # GroupNorm归一化
        x_norm = self.norm(x)

        # 估计动态参数 γ 和 β
        pooled = F.adaptive_avg_pool2d(x, 1)
        params = self.conv2(self.relu(self.conv1(pooled)))
        gamma, beta = torch.chunk(params, 2, dim=1)

        # 调制: Y = X̃ × (1+γ) + β
        out = x_norm * (1 + gamma) + beta
        return out


class ConvQKV(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Depthwise卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # Pointwise卷积生成Q、K、V
        self.pwconv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        # 位置编码卷积
        self.pos_conv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # Depthwise卷积
        x_dw = self.dwconv(x)

        # 通过Pointwise卷积生成Q、K、V
        qkv = self.pwconv(x_dw)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 添加位置编码
        pos = self.pos_conv(x)
        q = q + pos
        k = k + pos

        return q, k, v


class MultiScaleAttention(nn.Module):

    def __init__(self, dim, num_heads=8, scales=[1, 2, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scales = scales

       
        self.qkv_layers = nn.ModuleList([
            ConvQKV(dim, num_heads) for _ in scales
        ])

      
        self.fusion = nn.Conv2d(dim * len(scales), dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = []

        for i, scale in enumerate(self.scales):
           
            if scale > 1:
                x_scale = F.avg_pool2d(x, kernel_size=scale, stride=scale)
            else:
                x_scale = x

           
            q, k, v = self.qkv_layers[i](x_scale)

            _, _, H_s, W_s = q.shape
            N_s = H_s * W_s

            
            q = q.reshape(B, self.num_heads, self.head_dim, N_s)
            k = k.reshape(B, self.num_heads, self.head_dim, N_s)
            v = v.reshape(B, self.num_heads, self.head_dim, N_s)

           
            attn = torch.matmul(q.transpose(-2, -1), k) / \
                (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)

            
            out = torch.matmul(attn, v.transpose(-2, -1))
            out = out.transpose(-2, -1)

           
            out = out.reshape(B, C, H_s, W_s)

           
            if scale > 1:
                out = F.interpolate(out, size=(
                    H, W), mode='bilinear', align_corners=False)

            outputs.append(out)

       
        out = self.fusion(torch.cat(outputs, dim=1))
        return out


class DynamicConvFFN(nn.Module):
    def __init__(self, dim, num_experts=4, expansion_ratio=4):
        super().__init__()
        self.num_experts = num_experts
        hidden_dim = dim * expansion_ratio

       
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, num_experts, kernel_size=1),
            nn.Softmax(dim=1)
        )

        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, C, H, W = x.shape

        weights = self.router(x)  # B × E × 1 × 1

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))

        expert_outputs = torch.stack(
            expert_outputs, dim=1)  # B × E × C × H × W

        weights = weights.unsqueeze(2)  # B × E × 1 × 1 × 1
        output = (expert_outputs * weights).sum(dim=1)

        return output


class MSCABlock(nn.Module):

    def __init__(self, dim, num_heads=8, scales=[1, 2, 4], num_experts=4):
        super().__init__()
        self.norm1 = AdaptiveNorm(dim)
        self.attn = MultiScaleAttention(dim, num_heads, scales)
        self.norm2 = AdaptiveNorm(dim)
        self.ffn = DynamicConvFFN(dim, num_experts)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))
        return x


class MSCATransformer(nn.Module):

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_blocks=6,
                 num_heads=8,
                 scales=[1, 2, 4],
                 num_experts=4,
                 num_groups=32):
        super().__init__()


        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=1)


        self.blocks = nn.ModuleList([
            MSCABlock(dim, num_heads, scales, num_experts)
            for _ in range(num_blocks)
        ])


        self.output_proj = nn.Conv2d(dim, in_channels, kernel_size=1)


        self.global_residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):

        identity = x

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)

        output = identity + self.global_residual_weight * x

        return output


if __name__ == '__main__':
    output = []

    data_1 = torch.randn([1, 3, 256, 256]).to('cuda')
    model_0 = zongtimox()
    model_0.to('cuda')
    data_1 = model_0(data_1)
    print(data_1[0].shape, data_1[1].shape, data_1[2].shape)

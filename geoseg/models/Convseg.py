import torch
from torch import nn
from functools import partial
import torch.nn.functional as f
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import time
from torchvision.ops import deform_conv2d
# encoder 输出的本来就是四个特征图
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.): # 这个hidden feature 一般取的是in features 的4倍
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features) # 这个是做3×3卷积，然后能够得到位置信息, shape是保持不变的，并没有下采样
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights) # 从父类那里继承过来的方法，初始化参数用的

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads,qkv_bias = None,qk_scale=None,attn_drop = 0., proj_drop = 0.,sr_ratio = None ): # token的维度，QKV的维度都是一样的
#         super(Attention, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q = nn.Linear(dim,dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim , bias=qkv_bias)
#         self.v = nn.Linear(dim, dim , bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(dim, dim, 1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self,x,H,W):
#         B,N,C = x.shape
#         keys = self.k(x).reshape((B, self.dim, N))
#         queries = self.q(x).reshape((B, self.dim, N))
#         values = self.v(x).reshape((B, self.dim, N))
#         head_key_channels = self.head_dim
#         head_value_channels = self.head_dim
#
#         attended_values = []
#         for i in range(self.num_heads):
#             key = f.softmax(keys[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=2)
#             query = f.softmax(queries[
#                 :,
#                 i * head_key_channels: (i + 1) * head_key_channels,
#                 :
#             ], dim=1)
#             value = values[
#                 :,
#                 i * head_value_channels: (i + 1) * head_value_channels,
#                 :
#             ]
#             context = key @ value.transpose(1, 2)
#             attended_value = (
#                 context.transpose(1, 2) @ query
#             ).reshape(B, head_value_channels, H, W)
#             attended_values.append(attended_value)
#
#         aggregated_values = torch.cat(attended_values, dim=1)
#         reprojected_value = self.proj(aggregated_values)
#         attention = reprojected_value + x.reshape(B,C,H,W)
#
#         return attention.reshape(B,N,C)
class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x
class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
                # local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=5,
                #                        padding=2, stride=1, groups=dim // self.ca_num_heads)
                # local_conv = DeformableConv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=5,
                #                        padding=2, stride=1, groups=dim // self.ca_num_heads, offset_groups=dim // self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i]
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out, s_i], 2)
            s_out = s_out.reshape(B, C, H, W)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
            x = s_out * v

        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                          N).transpose(
                    1, 2)

        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x
    
# class Attention(nn.Module):
#     def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
#                  attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
#         super().__init__()

#         self.ca_attention = ca_attention
#         self.dim = dim
#         self.ca_num_heads = ca_num_heads
#         self.sa_num_heads = sa_num_heads

#         assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
#         assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

#         self.act = nn.GELU()
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.split_groups = self.dim // ca_num_heads

#         if ca_attention == 1:
#             self.v = nn.Linear(dim, dim, bias=qkv_bias)
#             self.s = nn.Linear(dim, dim, bias=qkv_bias)
#             for i in range(self.ca_num_heads):
#                 # local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
#                 #                        padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
#                 local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=5,
#                                        padding=2, stride=1, groups=dim // self.ca_num_heads)
#                 setattr(self, f"local_conv_{i + 1}", local_conv)
#             self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
#                                    groups=self.split_groups)
#             self.bn = nn.BatchNorm2d(dim * expand_ratio)
#             self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

#         else:
#             head_dim = dim // sa_num_heads
#             self.scale = qk_scale or head_dim ** -0.5
#             self.q = nn.Linear(dim, dim, bias=qkv_bias)
#             self.attn_drop = nn.Dropout(attn_drop)
#             self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#             self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         if self.ca_attention == 1:
#             v = self.v(x)
#             s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
#             for i in range(self.ca_num_heads):
#                 local_conv = getattr(self, f"local_conv_{i + 1}")
#                 s_i = s[i]
#                 s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
#                 if i == 0:
#                     s_out = s_i
#                 else:
#                     s_out = torch.cat([s_out, s_i], 2)
#             s_out = s_out.reshape(B, C, H, W)
#             s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
#             self.modulator = s_out
#             s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
#             x = s_out * v

#         else:
#             q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
#             kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
#             k, v = kv[0], kv[1]
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
#                 self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
#                                                                                                           N).transpose(
#                     1, 2)

#         # x = self.proj(x)
#         # x = self.proj_drop(x)

#         return x
    
class EAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class EAttention(nn.Module):
#     def __init__(self, dim, use_DropKey = True, mask_ratio =0.1,  num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.use_DropKey = use_DropKey
#         self.mask_ratio = mask_ratio
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         if self.use_DropKey == True:
#             m_r = torch.ones_like(attn) * self.mask_ratio
#             attn = attn + torch.bernoulli(m_r) * -1e12

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x


# 把attention和mlp叠在一起先，因为这个模块会重复很多遍，根据设置的参数depths决定
# 这个模块中用到了残差连接，但是论文中没有提出
class Block1(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1): # layer norm不同于batch norm，对于sequence来说layer norm的效果更好
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class Block2(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1): # layer norm不同于batch norm，对于sequence来说layer norm的效果更好
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

# 做词嵌入的
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]): # depths表示的是那个最基本的block有几层
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        '''
        这里做一点说明，这里写的patch_size的大小是7，stride=4，不像是VIT模型那样两个值都是patch-size，因为这篇文章提出的是带有overlap的embedded
        所以他会做padding，保证最后的patch——size就是预设的4×4
        '''
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0]) # 下采样四倍倍
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1]) #下采样两倍
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block1(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block1(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block2(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block2(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x) # H,W 是输入图像下采样之后的高宽 相乘以后就是词的个数
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W) # 这个时候输出的高宽是不会改变的，也就是N = H*W
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # 把那个sequence还原成图像，也就是得到了第一个feature map
        outs.append(x)                                             # 这里其实就是那个patch mergeing模块

        # stage 2
        x, H, W = self.patch_embed2(x) # 然后紧接着就跟上了embedding
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs  # 这里输出的是一个list，里面的四个元素就是四个不同尺度的feature map

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module): # 用3×3卷积实现位置编码
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # self.bn = nn.BatchNorm2d(dim)

    def forward(self, x, H, W): # 先把序列化成图像，然后再化成序列输出
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# class Decoder(nn.Module):
#     def __init__(self,dim_out_encoder=[32,64, 128, 256], decoder_dim=256, num_class=20, scale_factor = 2,batch_norm = nn.BatchNorm2d,active=nn.ReLU() ):
#         super(Decoder, self).__init__()

#         self.in_c1 = dim_out_encoder[0]
#         self.in_c2 = dim_out_encoder[1]
#         self.in_c3 = dim_out_encoder[2]
#         self.in_c4 = dim_out_encoder[3]
#         self.up_sample1 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
#         self.up_sample2 = nn.Upsample(scale_factor=scale_factor*2, mode='bilinear', align_corners=True)
#         self.up_sample3 = nn.Upsample(scale_factor=scale_factor*4, mode='bilinear', align_corners=True)
#         # self.up_sample4 = nn.Upsample(scale_factor=scale_factor*8, mode='bilinear', align_corners=True)
#         # self.up_sample5 = nn.Upsample(scale_factor=scale_factor*16,mode='bilinear', align_corners=True)

#         self.linear1 = nn.Conv2d(self.in_c1, decoder_dim, 1) # 用1*1卷积代替全连接层
#         self.linear2 = nn.Conv2d(self.in_c2, decoder_dim, 1)
#         self.linear3 = nn.Conv2d(self.in_c3, decoder_dim, 1)
#         self.linear4 = nn.Conv2d(self.in_c4, decoder_dim, 1)
#         self.linear_out = nn.Conv2d(decoder_dim*4, num_class, 1)

#         self.bn = batch_norm(dim_out_encoder[-1])
#         self.bn1 = batch_norm(num_class)
#         self.act = active


#     def forward(self,out_encoder):
# #         x3= self.act(self.bn(self.linear4(out_encoder[3])))

# #         # print(x3.shape)
# #         xd = self.up_sample3(x3)
# #         x2 = self.act(self.bn(self.linear3(out_encoder[2]))) + self.up_sample1(x3)
# #         xc = self.up_sample2(x2)
# #         x1 = self.act(self.bn(self.linear2(out_encoder[1]))) + self.up_sample1(x2)
# #         xb = self.up_sample1(x1)
# #         x0 = self.act(self.bn(self.linear1(out_encoder[0]))) + self.up_sample1(x1)
# #         # print(x0.shape)
# #         x = torch.cat((xd,xc,xb,x0),dim=1)
# #         # out_linear = [x1,x2,x3,x4]
# #         # x5 = self.fusion_feature(out_linear)
# #         # x = torch.cat((x1,x2,x3,x4,x5),dim=1)
# #         # x = self.act(self.bn1(self.linear_out(x)))
# #         # return self.up_sample2(x)
# #         x = self.act(self.bn1(self.linear_out(x)))
# #         return self.up_sample2(x)
#         x3 = self.up_sample1(out_encoder[3])
#         xd = self.up_sample3(out_encoder[3])
#         x2 = self.act(self.bn(self.linear3(out_encoder[2]))) + x3
#         xc = self.up_sample2(x2)
#         x1 = self.act(self.bn(self.linear2(out_encoder[1]))) + self.up_sample1(x2)
#         xb = self.up_sample1(x1)
#         x0 = self.act(self.bn(self.linear1(out_encoder[0]))) + self.up_sample1(x1)
#         x = torch.cat((xd,xc,xb,x0),dim=1)
#         x = self.act(self.bn1(self.linear_out(x)))
#         return self.up_sample2(x)


# 得到的四张特征图，然后最后拿来做分组的融合
# x1 x2 x3 x4 的尺寸是不一样的
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return f.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Groupfusing(nn.Module):
    def __init__(self, k_size=3, d_list=[1, 2, 5, 7], dim_out_encoder=[32, 64, 128, 256], decoder_dim=256, num_groups=4,
                 num_class=6, scale_factor=2,
                 batch_norm=nn.BatchNorm2d, active=nn.ReLU(), ):
        super(Groupfusing, self).__init__()
        self.in_c1 = dim_out_encoder[0]
        self.in_c2 = dim_out_encoder[1]
        self.in_c3 = dim_out_encoder[2]
        self.in_c4 = dim_out_encoder[3]
        self.num_groups = num_groups
        self.up_sample1 = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.up_sample2 = nn.Upsample(scale_factor=scale_factor * 2, mode='bilinear', align_corners=True)
        self.up_sample3 = nn.Upsample(scale_factor=scale_factor * 4, mode='bilinear', align_corners=True)

        # self.up_sample4 = nn.Upsample(scale_factor=scale_factor*8, mode='bilinear', align_corners=True)
        # self.up_sample5 = nn.Upsample(scale_factor=scale_factor*16,mode='bilinear', align_corners=True)

        self.linear1 = nn.Conv2d(self.in_c1, decoder_dim, 1)  # 用1*1卷积代替全连接层
        self.linear2 = nn.Conv2d(self.in_c2, decoder_dim, 1)
        self.linear3 = nn.Conv2d(self.in_c3, decoder_dim, 1)
        self.linear4 = nn.Conv2d(self.in_c4, decoder_dim, 1)
        self.linear_out = nn.Conv2d(decoder_dim * 2, num_class, 1)

        self.bn = batch_norm(dim_out_encoder[-1])
        self.bn1 = batch_norm(num_class)
        self.act = active
        group_size = decoder_dim // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            nn.Conv2d(group_size, group_size, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size)
        )

    def forward(self, out_encoder):
        x3 = self.up_sample1(out_encoder[3])
        # xd = self.up_sample3(out_encoder[3])
        x2 = self.act(self.bn(self.linear3(out_encoder[2]))) + x3
        xh = self.up_sample2(x2)
        # xc = self.up_sample2(x2)
        x1 = self.act(self.bn(self.linear2(out_encoder[1])))
        # xb = self.up_sample1(x1)
        xl = self.act(self.bn(self.linear1(out_encoder[0]))) + self.up_sample1(x1)
        print(x1.shape)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        xa = self.g0(torch.cat((xh[0], xl[0]), dim=1))
        xb = self.g1(torch.cat((xh[1], xl[1]), dim=1))
        xc = self.g2(torch.cat((xh[2], xl[2]), dim=1))
        xd = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((xa, xb, xc, xd), dim=1)
        # x = torch.cat((xd, xc, xb, x0), dim=1)
        x = self.act(self.bn1(self.linear_out(x)))
        return self.up_sample2(x)







class Segformer(nn.Module):
    def __init__(self,img_size=512, patch_size=16, in_chans=3, num_classes=20, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super(Segformer, self).__init__()
        self.encoder = MixVisionTransformer(img_size = img_size,patch_size = patch_size,
                                            in_chans=in_chans,num_classes=num_classes,embed_dims=embed_dims,
                                            num_heads = num_heads, mlp_ratios = mlp_ratios,qkv_bias = qkv_bias,
                                            qk_scale = qk_scale, drop_rate= drop_rate,attn_drop_rate=attn_drop_rate,
                                            drop_path_rate=drop_path_rate,norm_layer = norm_layer,
                                            depths = depths, sr_ratios=sr_ratios)
        self.decoder = Groupfusing(dim_out_encoder=embed_dims,decoder_dim=embed_dims[3],num_class=num_classes)

    def forward(self,img):
        out = self.encoder(img)
        out1 = self.decoder(out)
        return out1


if __name__ == '__main__':
    net = Segformer(img_size=512, patch_size=4, in_chans=3, num_classes=6, embed_dims=[ 40,80, 200,320 ],
                 num_heads=[4,4,4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3,4,6,4], sr_ratios=[8, 4, 2, 1])
    # net = Segformer(img_size=256, patch_size=4, in_chans=12, num_classes=2, embed_dims=[ 40, 80, 200 ,320 ],
    #              num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0,
    #              attn_drop_rate=0, drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #              depths=[3,4,6,4], sr_ratios=[8, 4, 2, 1])
    # net = Segformer(img_size=512, patch_size=4, in_chans=3, num_classes=6, embed_dims=[ 32,64, 128,256 ],
    #              num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #              attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    #              depths=[2,2,2,2], sr_ratios=[8, 4, 2, 1])
    # net = Segformer(img_size=512, patch_size=4, in_chans=3, num_classes=6, embed_dims=[ 40,80, 160,320 ],
    #              num_heads=[4, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #              attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
    #              depths=[3,4,6,4], sr_ratios=[8, 4, 2, 1])
    from speed import speed
    speed(net)

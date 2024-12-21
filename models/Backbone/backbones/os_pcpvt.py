import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import Attention as TimmAttention
from mmcv.runner import (auto_fp16, force_fp32,)

import copy
from collections import OrderedDict


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


class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1, sr_ratio=1.0):
        """
        ws 1 for stand attention
        """
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    @auto_fp16()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads,
                                            C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
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

    @auto_fp16()
    def forward(self, x, H_x, W_x,H_z, W_z):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            z, x = torch.split(x, [H_z * W_z, H_x * W_x], dim=1)  # 考虑是否添加 contiguous()

            x_ = x.permute(0, 2, 1).reshape(B, C, H_x, W_x).contiguous()
            z_ = z.permute(0, 2, 1).reshape(B, C, H_z, W_z).contiguous()

            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            z_ = self.sr(z_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            z_ = self.norm(z_)
            x_qsize = x_.size()[1]
            z_qsize = z_.size()[1]
            x_ = torch.cat([z_, x_], dim=1)

            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            x_qsize = H_x * W_x
            z_qsize = H_z * W_z
        k, v = kv[0], kv[1]

        q_z, q_x = torch.split(q, [H_z * W_z, H_x * W_x], dim=2)  # 考虑是否添加 contiguous()
        k_z, k_x = torch.split(k, [z_qsize,x_qsize], dim=2)
        v_z, v_x = torch.split(v, [z_qsize,x_qsize], dim=2)

        ################################ all_attention########################################################
        # attn = (q_z @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn_z = (attn @ v).transpose(1, 2).reshape(B, H_z * W_z, C).contiguous()
        #
        # attn = (q_x @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn_x = (attn @ v).transpose(1, 2).reshape(B, H_x * W_x, C).contiguous()
        ################################################################################################################
        ###################################### mixattention ############################################################
        attn = (q_z @ k_z.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_z = (attn @ v_z).transpose(1, 2).reshape(B, H_z * W_z, C).contiguous()

        attn = (q_x @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_x = (attn @ v).transpose(1, 2).reshape(B, H_x * W_x, C).contiguous()
        ################################################################################################################

        ###################################### xxxattention ############################################################
        # attn = (q_z @ k_x.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn_z = (attn @ v_x).transpose(1, 2).reshape(B, H_z * W_z, C).contiguous()
        #
        # attn = (q_x @ k_z.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn_x = (attn @ v_z).transpose(1, 2).reshape(B, H_x * W_x, C).contiguous()
        ################################################################################################################
        x = torch.cat([attn_z, attn_x], dim=1)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H_x, W_x,H_z, W_z):
        x = x + self.drop_path(self.attn(self.norm1(x), H_x, W_x,H_z, W_z))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer, norm_layer)

    def forward(self, x, H, W):
        return super(SBlock, self).forward(x)


class GroupBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(PatchEmbed(img_size // patch_size // 2**(i-1), 2, embed_dims[i-1], embed_dims[i]))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) -1 else self.patch_embeds[-1].num_patches
            self.pos_embeds.append(nn.Parameter(torch.zeros(1, patch_num, embed_dims[i])))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k])
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        # self.norm = norm_layer(embed_dims[-1])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        # classification head
        # self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.blocks[k][i].drop_path.drop_prob = dpr[cur + i]
            cur += self.depths[k]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @auto_fp16()
    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim),)
        self.s = s

    @auto_fp16()
    def forward(self, x, H_x, W_x,H_z, W_z):
        B, N, C = x.shape
        z, x = torch.split(x, [H_z * W_z, H_x * W_x], dim=1)  # 考虑是否添加 contiguous()

        feat_token_x = x
        feat_token_z = z
        cnn_feat_x = feat_token_x.transpose(1, 2).view(B, C, H_x, W_x).contiguous()
        cnn_feat_z = feat_token_z.transpose(1, 2).view(B, C, H_z, W_z).contiguous()
        if self.s == 1:
            x = self.proj(cnn_feat_x) + cnn_feat_x
            z = self.proj(cnn_feat_z) + cnn_feat_z
        else:
            x = self.proj(cnn_feat_x)
            z = self.proj(cnn_feat_z)
        x = x.flatten(2).transpose(1, 2).contiguous()
        z = z.flatten(2).transpose(1, 2).contiguous()
        x = torch.cat([z, x], dim=1)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class CPVTV2(PyramidVisionTransformer):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block, F4=False,pretrained="/media/zeh/4d723c17-52ed-4771-9ff5-c5b4cf1675e9/cjh/FPI_backbone/pretrain_model/pcpvt_small.pth"):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths,
                                     sr_ratios, block_cls)
        self.F4 = F4
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.ModuleList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims]
        )
        self.apply(self._init_weights)
        # self.init_weights(pretrained)

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        import math
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
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    @auto_fp16()
    def forward_features(self, x,z):  # z: small x: big
        outputs = {"x":[],"z":[]}

        B = x.shape[0]

        for i in range(len(self.depths)):  # z: small x: big
            x, (H_x, W_x) = self.patch_embeds[i](x)  # B HW C BIG
            z, (H_z, W_z) = self.patch_embeds[i](z)  # B HW C SMALL
            x = torch.cat((z, x), dim=1)
            x = self.pos_drops[i](x)

            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H_x, W_x,H_z, W_z)
                if j == 0:
                    x = self.pos_block[i](x, H_x, W_x,H_z, W_z)
            z, x = torch.split(x, [H_z * W_z, H_x * W_x], dim=1)  # 考虑是否添加 contiguous()
            x = x.reshape(B, H_x, W_x, -1).permute(0, 3, 1, 2).contiguous()
            z = z.reshape(B, H_z, W_z, -1).permute(0, 3, 1, 2).contiguous()
            outputs["x"].append(x)
            outputs["z"].append(z)

        return outputs

    def forward(self, x,z): # z: small x: big
        x = self.forward_features(x,z)

        # if self.F4:
        #     x = x[3:4]

        return x


class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=SBlock, F4=False):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls, F4)


class ALTGVT(PCPVT):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=GroupBlock, wss=[7, 7, 7],
                 F4=False, extra_norm=False):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls, F4)
        del self.blocks
        self.wss = wss
        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(norm_layer(dim))
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])])
            self.blocks.append(_block)
        self.apply(self._init_weights)

    @auto_fp16()
    def forward_features(self, x):
        outputs = list()

        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    # We use a very simple implementation of PEG (single Sep at each stage).
                    # In fact, we can insert many PEGs into each block, which
                    # can improve the performance further. The core idea behind PEG is that even simple depthwise
                    # convolution provides sufficient positional information if paddings is given. It works quite well
                    # when global reception field is given. See https://arxiv.org/abs/2102.10882 for more details.
                    x = self.pos_block[i](x, H, W)
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)

        return outputs


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict




class os_pcpvt_small(CPVTV2):
    def __init__(self, **kwargs):
        # super(pcpvt_small, self).__init__(
        #     patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        #     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        #     drop_rate=0.0, drop_path_rate=0.2,pretrained="/media/zeh/2TBlue/FPI/pretrain_model/pcpvt_small.pth")
        super(os_pcpvt_small, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320], num_heads=[1, 2, 5], mlp_ratios=[8, 8, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6], sr_ratios=[8, 4, 2],
            drop_rate=0.0, drop_path_rate=0.2,pretrained="/media/zeh/2TBlue/FPI/pretrain_model/pcpvt_small.pth")


    def load_param(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def load_param_self_backbone(self, pretrained=None):
        if isinstance(pretrained, str):
            model_dict= self.state_dict()
            state_dict = torch.load(pretrained)
            state_dict = {k.split("model_uav.transformer.")[-1]: v for k,v in state_dict.items() if (k.split("model_uav.transformer.")[-1] in model_dict )}
            self.load_state_dict(state_dict)


class os_pcpvt_base(CPVTV2):
    def __init__(self, **kwargs):
        super(os_pcpvt_base, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.2,pretrained="/media/xxx/pretrain_model/pcpvt_base.pth")


class os_pcpvt_large(CPVTV2):
    def __init__(self, **kwargs):
        super(os_pcpvt_large, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.2)


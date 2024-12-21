from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
from .Backbone.backbones.vit_pytorch import Block

__all__ = ['SiamFC_HEAD', 'FeatureFusion']


class SwinTrack(nn.Module):
    def __init__(self, opt):
        super(SwinTrack, self).__init__()
        original_dim = 384
        opt.dim = 48
        self.linear1 = nn.Linear(original_dim, opt.dim)
        self.linear2 = nn.Linear(original_dim, opt.dim)
        self.z_patches = opt.UAVhw[0] // 16 * opt.UAVhw[1] // 16
        self.x_patches = opt.Satellitehw[0] // 16 * opt.Satellitehw[1] // 16
        num_patches = self.z_patches + self.x_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, opt.dim))
        self.attnBlock = nn.Sequential(Block(dim=opt.dim, num_heads=12))
        self.out_norm = nn.LayerNorm(opt.dim)
        # cls and loc
        self.cls_linear = nn.Linear(opt.dim, 1)
        self.loc_linear = nn.Linear(opt.dim, 2)

    def forward(self, z, x):
        z_feat = self.linear1(z)
        x_feat = self.linear2(x)
        concat_feature = torch.concat((z_feat, x_feat), dim=1)
        concat_feature += self.pos_embed
        out_feature = self.attnBlock(concat_feature)[:, self.z_patches:, :]
        out_feature = self.out_norm(out_feature)
        cls_feat = self.cls_linear(out_feature)
        # loc_feat = self.loc_linear(decoder_feat)

        return cls_feat  # B*1*25*25 B*2*25*25


class SiamFC_HEAD(nn.Module):
    def __init__(self, out_scale=0.01):
        super(SiamFC_HEAD, self).__init__()
        self.out_scale = out_scale
        # self.bbb = nn.BatchNorm2d(1)
        # self.drop = nn.Dropout(0.1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        # out = self.drop(out)
        # out = self.bbb(out)
        return out


class SiamFC_HEAD_tree(nn.Module):
    def __init__(self, out_scale=0.01):
        super(SiamFC_HEAD_tree, self).__init__()
        self.out_scale = out_scale
        loc_output = 2

        self.temple_loc_conv = nn.Conv2d(64, 64 * loc_output, kernel_size=1)
        self.search_loc_conv = nn.Conv2d(64, 64, kernel_size=1)
        # self.bbb = nn.BatchNorm2d(1)
        # self.drop = nn.Dropout(0.1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x)

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        temple = self.temple_loc_conv(z)
        search = self.search_loc_conv(x)

        x = x.view(-1, nz * c, h, w)
        out_cls = F.conv2d(x, z, groups=nz, padding=zh // 2)
        out_cls = out_cls.view(nx, -1, out_cls.size(-2), out_cls.size(-1))


        search = search.view(-1, nz * c, h, w)
        temple = temple.view(-1, c, temple.size()[2], temple.size()[3])
        out_loc = F.conv2d(search, temple, groups=nz, padding=zh // 2)
        out_loc = out_loc.view(nx, -1, out_loc.size(-2), out_loc.size(-1))

        return out_cls * self.out_scale, out_loc * self.out_scale



class SiamFC_HEAD_loc(nn.Module):
    def __init__(self, out_scale=0.01):
        super(SiamFC_HEAD_loc, self).__init__()
        self.out_scale = out_scale
        loc_output = 2

        self.temple_loc_conv = nn.Conv2d(64, 64 * loc_output, kernel_size=1)
        self.search_loc_conv = nn.Conv2d(64, 64, kernel_size=1)
        # self.bbb = nn.BatchNorm2d(1)
        # self.drop = nn.Dropout(0.1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x)

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        temple = self.temple_loc_conv(z)
        search = self.search_loc_conv(x)

        search = search.view(-1, nz * c, h, w)
        temple = temple.view(-1, c, temple.size()[2], temple.size()[3])
        out_loc = F.conv2d(search, temple, groups=nz, padding=zh // 2)
        out_loc = out_loc.view(nx, -1, out_loc.size(-2), out_loc.size(-1))

        return out_loc * self.out_scale


class SiamFC_HEAD_dweasy(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SiamFC_HEAD_dweasy, self).__init__()
        self.out_scale = out_scale
        self.dwconv_z = nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64)
        self.dwconv_x = nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64)
        self.act = nn.GELU()
        self.drop = nn.Dropout()

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        x = x + self.dwconv_x(x)
        z = z + self.dwconv_z(z)
        x = self.act(x)
        z = self.act(z)
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class SiamFC_HEAD_dw(nn.Module):
    def __init__(self, out_scale=0.001, dim=64):
        super(SiamFC_HEAD_dw, self).__init__()
        self.out_scale = out_scale
        self.conv0_x = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_x = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_x = nn.Conv2d(dim, dim, 1)
        self.conv0_z = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_z = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_z = nn.Conv2d(dim, dim, 1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def dw_x(self, x):
        u = x.clone()
        attn = self.conv0_x(x)
        attn = self.conv_spatial_x(attn)
        attn = self.conv1_x(attn)
        return u + attn

    def dw_z(self, x):
        u = x.clone()
        attn = self.conv0_z(x)
        attn = self.conv_spatial_z(attn)
        attn = self.conv1_z(attn)
        return u + attn

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        z = self.dw_z(z)
        x = self.dw_x(x)
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class SiamFC_HEAD_depwise(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size, out_channels, out_scale=0.001, dim=64):
        super(SiamFC_HEAD_depwise, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def dw_z(self, x):
        u = x.clone()
        attn = self.conv0_z(x)
        attn = self.conv_spatial_z(attn)
        attn = self.conv1_z(attn)
        return u + attn

    def _fast_xcorr(self, kernel, search):
        """depthwise cross correlation
        """
        zh = kernel.size()[2]
        batch = kernel.size(0)
        channel = kernel.size(1)
        search = search.view(1, batch * channel, search.size(2), search.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(search, kernel, groups=batch * channel, padding=zh // 2)
        out = out.view(batch, channel, out.size(2), out.size(3))
        # out = self.dw_z(out)
        out = self.proj_2(out)
        return out


class SiamFC_HEAD_pixel_face_correlation(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SiamFC_HEAD_pixel_face_correlation, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        kernel = z
        feature = x
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))

        b, c, h, w = feature.shape
        ker = kernel.reshape(b, c, -1).transpose(1, 2)
        feat = feature.reshape(b, c, -1)
        corr = torch.matmul(ker, feat)
        out = corr.reshape(*corr.shape[:2], h, w)

        return out


class SiamFC_HEAD_pixel_to_global(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SiamFC_HEAD_pixel_to_global, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, kernel, search):
        """depthwise cross correlation
        """
        b, c, h, w = search.shape
        ker1 = kernel.reshape(b, c, -1)
        ker2 = ker1.transpose(1, 2)
        feat = search.reshape(b, c, -1)
        S1 = torch.matmul(ker2, feat)
        S2 = torch.matmul(ker1, S1)
        out = S2.reshape(*S2.shape[:2], h, w)
        return out


class SiamFC_HEAD_normal(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SiamFC_HEAD_normal, self).__init__()
        self.out_scale = out_scale
        self.norm = nn.LayerNorm(1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        out = out.view(nx, out.size(-2) * out.size(-1), -1)
        out = self.norm(out)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class ResBlock2(nn.Module):
    def __init__(self, input_feature, planes, dilated=1, group=1):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(input_feature, planes, kernel_size=1, bias=False, groups=group)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1 * dilated, bias=False, dilation=dilated,
                               groups=group)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, input_feature, kernel_size=1, bias=False, groups=group)
        self.bn3 = nn.InstanceNorm2d(input_feature)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class SiamFC_HEADconver1(nn.Module):
    def __init__(self, out_scale=0.001, size=196):
        super(SiamFC_HEADconver1, self).__init__()
        self.out_scale = out_scale
        # self.cls = self.make_layer2(196, 1)
        # self.cls = nn.Conv2d(196, 1, kernel_size=1, bias=False)
        self.Ranking = nn.Sequential(self.make_layer2(size, 128), ResBlock2(128, 32, 2), self.make_layer2(128, 1))

    def to_kernel(self, feature):
        size = feature.size()
        return feature.view(size[1], size[2] * size[3]).transpose(0, 1).unsqueeze(2).unsqueeze(3).contiguous()

    def correlate(self, Kernel, Feature):
        corr = torch.nn.functional.conv2d(Feature, Kernel, stride=1)
        return corr

    def make_layer2(self, input_feature, out_feature, up_scale=1, ksize=3, d=1, groups=1):
        p = int((ksize - 1) / 2)
        if up_scale == 1:
            return nn.Sequential(
                nn.InstanceNorm2d(input_feature),
                nn.ReLU(),
                nn.Conv2d(input_feature, out_feature, ksize, padding=p, dilation=d, groups=groups),
            )
        return nn.Sequential(
            nn.InstanceNorm2d(input_feature),
            nn.ReLU(),
            nn.Conv2d(input_feature, out_feature, ksize, padding=p),
            nn.UpsamplingBilinear2d(scale_factor=up_scale),
        )

    def forward(self, z, x):
        return self._fast_xcorr(z, x)

    def _fast_xcorr(self, Kernel_tmp, Feature):
        for idx in range(len(Feature)):
            ker = Kernel_tmp[idx: idx + 1]
            feature = Feature[idx: idx + 1]
            ker_R = self.to_kernel(ker)
            corr_R = self.correlate(ker_R, feature)
            R_map = self.Ranking(corr_R)
            # R_map = F.relu(self.cls(corr_R))
            # Ranking attention scores
            # T_corr = F.max_pool2d(corr_R, 2)
            # T_corr = T_corr.view(-1, 196, 625)
            # T_corr = T_corr.transpose(1, 2).view(-1, 196, 25, 25)
            # R_map = F.relu(self.cls(corr_R)
            if idx == 0:
                feature_out = R_map
            else:
                feature_out = torch.concat((feature_out, R_map), 0)

        return feature_out


class SiamFC_HEAD_rpn(nn.Module):
    def __init__(self, out_scale=0.001):
        super(SiamFC_HEAD_rpn, self).__init__()
        self.out_scale = out_scale
        self.cls_64 = nn.Conv2d(64, 64 * 3, kernel_size=1, stride=1)
        self.cls_128 = nn.Conv2d(128, 128 * 10, kernel_size=1, stride=1)
        self.cls_256 = nn.Conv2d(320, 320 * 10, kernel_size=1, stride=1)

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        zn, zc, zh, zw = z.size()
        if zc == 64:
            z = self.cls_64(z)
        elif zc == 128:
            z = self.cls_128(z)
        elif zc == 320:
            z = self.cls_256(z)
        z = z.view(-1, zc, zh, zw)
        nx, c, h, w = x.size()
        x = x.view(-1, zn * c, h, w)
        out = F.conv2d(x, z, groups=zn, padding=zh // 2)
        # out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out


class cross_HEAD(nn.Module):
    def __init__(self, opt, out_scale=0.001):
        super(cross_HEAD, self).__init__()
        self.cross_attn1 = nn.MultiheadAttention(1, 1, dropout=0.0)
        self.cross_attn2 = nn.MultiheadAttention(1, 1, dropout=0.0)
        self.cross_attn3 = nn.MultiheadAttention(1, 1, dropout=0.0)
        self.conv1 = nn.Conv2d(64, 1, 1, 1)
        self.conv2 = nn.Conv2d(128, 1, 1, 1)
        self.conv3 = nn.Conv2d(320, 1, 1, 1)
        self.conv11 = nn.Conv2d(64, 1, 1, 1)
        self.conv22 = nn.Conv2d(128, 1, 1, 1)
        self.conv33 = nn.Conv2d(320, 1, 1, 1)

    def forward(self, z, x):
        b, c, h, w = x.shape
        if c == 64:
            z = self.conv1(z)
            x = self.conv11(x)
        if c == 128:
            z = self.conv2(z)
            x = self.conv22(x)
        if c == 320:
            z = self.conv3(z)
            x = self.conv33(x)
        b, c, h, w = x.shape
        k = v = z.view(b, c, -1).permute(2, 0, 1).contiguous()
        q = x.view(b, c, -1).permute(2, 0, 1).contiguous()
        if c == 64:
            k1 = self.cross_attn1(query=q, key=k, value=v)[0]  # 25*25，B，384
        if c == 128:
            k1 = self.cross_attn2(query=q, key=k, value=v)[0]  # 25*25，B，384
        if c == 320:
            k1 = self.cross_attn3(query=q, key=k, value=v)[0]  # 25*25，B，384
        if c == 1:
            k1 = self.cross_attn3(query=q, key=k, value=v)[0]  # 25*25，B，384
        out_feature = k1.permute(1, 2, 0).view(b, c, h, w)

        return out_feature


class cross_HEAD1(nn.Module):
    def __init__(self, opt, out_scale=0.001):
        super(cross_HEAD1, self).__init__()
        self.cross_attn1 = nn.MultiheadAttention(1, 1, dropout=0.0)
        self.conv1 = nn.Conv2d(384, 1, 1, 1)
        self.conv11 = nn.Conv2d(384, 1, 1, 1)
        self.norm1 = nn.LayerNorm(1)

    def forward(self, z, x):
        z = self.conv1(z)
        x = self.conv11(x)
        b, c, h, w = x.shape
        k = v = z.view(b, c, -1).permute(2, 0, 1).contiguous()
        q = x.view(b, c, -1).permute(2, 0, 1).contiguous()
        k = v = self.norm1(k)
        q = self.norm1(q)
        k1 = self.cross_attn1(query=q, key=k, value=v)[0]  # 25*25，B，384
        out_feature = k1.permute(1, 2, 0).view(b, c, h, w)
        return out_feature


class SiamRPN_HEAD(nn.Module):
    def __init__(self, out_scale=0.001, in_channels=64):
        super(SiamRPN_HEAD, self).__init__()
        self.out_scale = out_scale
        loc_output = 2

        self.template_cls_conv = nn.Conv2d(in_channels,
                                           in_channels * loc_output, kernel_size=1)

        self.search_cls_conv = nn.Conv2d(in_channels,
                                         in_channels, kernel_size=1)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)
        self.norm = nn.BatchNorm2d(loc_output)

    def forward(self, z, x):
        template = self.template_cls_conv(z)
        search = self.search_cls_conv(x)
        cls = self._rpn_fast_xcorr(z, x) * self.out_scale
        loc = self._rpn_fast_xcorr(template, search)
        loc = self.loc_adjust(loc) * self.out_scale
        # loc = self.norm(loc)
        return cls, loc

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz, zc, zh, zw = z.size()
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz, padding=zh // 2)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

    def _rpn_fast_xcorr(self, kernel, x):
        """group conv2d to calculate cross correlation, fast version
            """
        batch = kernel.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])  # [b,384,7,7]
        px = x.view(1, -1, x.size()[2], x.size()[3])  # [1,2*384,25,25]
        po = F.conv2d(px, pk, groups=batch, padding=kernel.size()[2] // 2)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po


class FeatureFusion(nn.Module):
    def __init__(self, opt):
        super(FeatureFusion, self).__init__()
        if opt.backbone == "Deit-S":
            ndim = 384
        elif opt.backbone == "Vit-S":
            ndim = 768
        elif opt.backbone == "pcpvt_small":
            ndim = 1
        else:
            raise NameError("!!!!!!!")
        self.cross_attn1 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.cross_attn2 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.cross_attn3 = nn.MultiheadAttention(ndim, 8, dropout=0.0)
        self.norm1 = nn.LayerNorm(ndim)
        self.norm2 = nn.LayerNorm(ndim)
        self.norm3 = nn.LayerNorm(ndim)

        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(ndim, 256)

        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 256)

        self.activation3 = nn.GELU()
        self.dropout3 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, 64)

        self.proj = nn.Linear(64, 1)

    def forward(self, z, x):
        k = v = z.transpose(0, 1).contiguous()
        q = x.transpose(0, 1).contiguous()
        k1 = self.cross_attn1(query=k, key=q, value=q)[0] + v  # 25*25，B，384
        k1 = v1 = self.norm1(k1)

        src1 = self.cross_attn2(query=q, key=k1, value=v1)[0] + q  # 25*25，B，384
        src1 = self.norm2(src1)
        src2 = self.cross_attn3(query=src1, key=src1, value=src1)[0] + src1
        src2 = self.norm3(src2)
        src2 = src2.transpose(0, 1).contiguous()

        res1 = self.dropout1(self.activation1(self.linear1(src2)))
        res2 = self.dropout2(self.activation2(self.linear2(res1))) + res1
        res = self.dropout3(self.activation3(self.linear3(res2)))
        res = self.proj(res)
        return res

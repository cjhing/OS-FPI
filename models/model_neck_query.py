"""
-*- coding: utf-8 -*-

@作者(Author) : Chen Jiahao
@时间(Time) : 2023/3/1 22:47
@File : model_neck_query.py
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from .neck import PyramidFeatures, ASPP
from .head import SiamFC_HEAD,SiamFC_HEAD_tree,SiamFC_HEAD_rpn,SiamRPN_HEAD,SiamFC_HEAD_loc

class deit_GC(nn.Module):
    def __init__(self):
        super(deit_GC, self).__init__()
        self.head = SiamFC_HEAD()


    def vector2array(self, vector):
        n, p, c = vector.shape
        h = w = np.sqrt(p)
        if int(h) * int(w) != int(p):
            raise ValueError("p can not be sqrt")
        else:
            h = int(h)
            w = int(w)
        array = vector.permute(0, 2, 1).contiguous().view(n, c, h, w)
        return array

    def forward(self, feature_in, opt=None):
        x = feature_in["x"]
        z = feature_in["z"]

        z = self.vector2array(z)
        x = self.vector2array(x)
        cls = self.head(z, x)
        cls = F.interpolate(cls, size=opt.Satellitehw, mode='nearest')
        return cls, None

class GC(nn.Module):
    def __init__(self):
        super(GC, self).__init__()
        self.head = SiamFC_HEAD()

    def forward(self, feature_in, opt=None):
        x = feature_in["x"][2]
        z = feature_in["z"][2]
        cls = self.head(z, x)
        cls = F.interpolate(cls, size=opt.Satellitehw, mode='nearest')
        return cls, None

class WAMF(nn.Module):
    def __init__(self):
        super(WAMF, self).__init__()
        self.FPN = PyramidFeatures(64, 128, 320, feature_size=64)  # 1,1,1
        self.FPN2 = PyramidFeatures(64, 128, 320, feature_size=64)  # 1,1,1
        self.head = SiamFC_HEAD()
        self.w1 = nn.Parameter(torch.ones(3))

    def forward(self, feature_in, opt=None):
        x = feature_in["x"][0:3]
        z = feature_in["z"][0:3]
        w1 = self.w1
        w1 = w1 / w1.sum()
        freature_list = []
        x = self.FPN(x)[0]
        z = self.FPN2(z)
        for num, t in enumerate(z):
            cls = self.head(t, x)
            freature_list.append(cls)
        cls = w1[0] * freature_list[0] + w1[1] * freature_list[1] + w1[2] * freature_list[2]
        cls = F.interpolate(cls, size=opt.Satellitehw, mode='nearest')
        return cls, None

class X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy(nn.Module):
    def __init__(self):
        super(X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy, self).__init__()
        self.FPN = PyramidFeatures(64, 128, 320, feature_size=128)  # 1,1,1
        self.aspp = ASPP(in_channels=128, atrous_rates=[12, 24, 32], out_channels=64)
        self.head = SiamFC_HEAD()
        self.project_xy = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1, bias=False),
        )

    def forward(self, feature_in, opt=None):
        x = feature_in["x"][0:3]
        z = feature_in["z"][0:3]
        x = self.FPN(x)[0]  # big
        x = self.aspp(x)
        loc = self.project_xy(x)
        cls = self.head(z[0], x)
        cls = F.interpolate(cls, size=opt.Satellitehw, mode='nearest')
        loc = F.interpolate(loc, size=opt.Satellitehw, mode='nearest')
        return cls, loc

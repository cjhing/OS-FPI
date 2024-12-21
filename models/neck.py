import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(self.conv1(inputs2))
        outputs = inputs2 +inputs1
        # outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self,  C3_size, C4_size, C5_size, feature_size=384):
        super(Unet, self).__init__()
        in_filters = [64, 128, 320]
        out_filters = [64, 64, 128]
        # upsampling
        self.up_concat2 = unetUp(in_filters[2], out_filters[2])
        # self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[1], out_filters[1])

        # final conv (without any concat)
        # self.final=nn.Conv2d(out_filters[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(out_filters[0], feature_size, 1)

    def forward(self, inputs):
        feat1, feat2, feat3 = inputs

        up2 = self.up_concat2(feat2, feat3)
        final = self.up_concat1(feat1, up2)

        # final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()



class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=384):  # 384 1
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
    #     self.P7_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     import math
    #     if isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1.0)
    #         m.bias.data.zero_()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # P6_x = self.P6(C5)
        # # panet
        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x]


class PyramidFeatures2(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=384):  # 384 1
        super(PyramidFeatures2, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x]

class PANET_fusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=1):  # 384 1
        super(PANET_fusion, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(1, 1, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class PANET(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=384):  # 384 1
        super(PANET, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(feature_size, feature_size, 3, 2, 1)
        # self.convN2 = nn.Conv2d(feature_size, feature_size, 3, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class PANET_conver(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=300):  # 384 1
        super(PANET_conver, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_c = nn.Conv2d(feature_size * 2, feature_size, kernel_size=1, stride=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_c = nn.Conv2d(feature_size * 2, feature_size, kernel_size=1, stride=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.convN = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.gelu = nn.GELU()

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = torch.concat((P5_upsampled_x, P4_x), 1)
        P4_x = self.P4_c(P4_x)
        # P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)  # be replase by P4_c

        P3_x = self.P3_1(C3)
        P3_x = torch.concat((P4_upsampled_x, P3_x), 1)
        P3_x = self.P3_c(P3_x)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)  # be replase by P4_c

        P6_x = self.P6(C5)
        # panet
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        N2 = P3_x
        N2_ = self.convN(N2)  # down samplle
        N2_ = self.relu(N2_)
        #
        N3 = N2_ + P4_x
        #
        N3_ = self.convN(N3)  # down samplle
        N3_ = self.relu(N3_)

        N4 = N3_ + P5_x
        return [N2, N3, N4]


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[3, 5, 7]):
        super(SpatialPyramidPooling, self).__init__()
        self.conv1 = nn.Conv2d(320, 160, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(640, 320, kernel_size=1, stride=1, bias=False)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    # def conv2d(self, filter_in, filter_out, kernel_size, stride=1):
    #     pad = (kernel_size - 1) // 2 if kernel_size else 0
    #     return nn.Sequential(OrderedDict([
    #         ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
    #         ("bn", nn.BatchNorm2d(filter_out)),
    #         ("relu", nn.LeakyReLU(0.1)),
    #     ]))
    #
    # def make_three_conv(self, filters_list, in_filters):
    #     m = nn.Sequential(
    #         self.conv2d(in_filters, filters_list[0], 1),
    #         self.conv2d(filters_list[0], filters_list[1], 3),
    #         self.conv2d(filters_list[1], filters_list[0], 1),
    #     )
    #     return m

    def forward(self, z):
        x = self.conv1(z[2])
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        # features = torch.cat(features + [x], dim=1)
        features = features[0]+features[1]+features[2]+x
        z[2] = features

        return z



class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates, out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [            # add one normal conv
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:   # add three dilation conv
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)





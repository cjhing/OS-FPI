import torch.nn as nn
from .Backbone import make_transformer_model, make_cnn_model
from .head import SiamFC_HEAD
import numpy as np
from .model_neck_query import GC, WAMF, X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy
import torch

Transformer_model_list = ["Deit-S", "Vit-S", "Swin-Transformer-S", "os_pcpvt_small"]
CNN_model_list = ["Resnet50", "Resnest50"]


class SiamUAV_Transformer_Model(nn.Module):
    def __init__(self, opt):
        super(SiamUAV_Transformer_Model, self).__init__()
        backbone = opt.backbone
        self.model_uav = make_transformer_model(opt, opt.UAVhw, transformer_name=backbone)
        # self.model_neck = X_FPN_ASPP_conv_12()  # 1,1,1
        if opt.neck == "GC":
            self.model_neck = GC()
        elif opt.neck == 'WAMF':
            self.model_neck = WAMF()
        elif opt.neck == "X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy":
            self.model_neck = X_FPN_ASPP_conv_12_24_32_HEAD_Z0_xy()

        self.opt = opt

    def forward(self, z, x):
        out = self.model_uav(x, z)
        cls, loc = self.model_neck(out, opt=self.opt)
        return cls, loc

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

    def get_part(self, x, padding_time):
        h, w = x.shape[-2:]
        cx, cy = h // 2, w // 2
        ch, cw = h // (padding_time + 1) // 2, w // (padding_time + 1) // 2
        x1, y1, x2, y2 = cx - ch, cy - cw, cx + ch + 1, cy + cw + 1
        part = x[:, :, int(x1):int(x2), int(y1):int(y2)]
        return part

    def load_params(self, opt):
        # load pretrain param
        if opt.backbone == "PVT-T":
            pretrain = "pretrain_model/pvt_tiny.pth"
        if opt.backbone == "Vit-S":
            pretrain = "pretrain_model/vit_small_p16_224-15ec54c9.pth"
        if opt.backbone == "Deit-S":
            pretrain = "pretrain_model/deit_small_distilled_patch16_224-649709d9.pth"
        if opt.backbone == "Swin-Transformer-S":
            pretrain = "pretrain_model/swin_small_patch4_window7_224.pth"
        if opt.backbone == "os_pcpvt_small":
            pretrain = r"D:\OS-PFI\net_040.pth"

        # if not opt.share:
        #     self.model_satellite.transformer.load_param(pretrain)
        if opt.USE_old_model:
            # pretran_model = torch.load(opt.checkpoints)
            # model2_dict = self.state_dict()
            # state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys()}
            # model2_dict.update(state_dict)
            # self.load_state_dict(model2_dict)

            # state_dict = torch.load(opt.checkpoints)
            # self.load_state_dict(state_dict)
            # self.load_param_self_backbone("/media/zeh/2TBlue/FPI/pretrain_model/OS_PCPVT_77.pth")
            # self.model_uav.transformer.load_param_self_backbone("/media/zeh/2TBlue/FPI/pretrain_model/double_aspp_24_75.pth")

            # self.model_uav.transformer.load_param_self_backbone(
            #     "/media/zeh/2TBlue/FPI/pretrain_model/OS_PCPVT_ASPP_74.pth")

            # self.model_uav.transformer.load_param_self_backbone(
            #     "/media/zeh/2TBlue/FPI/pretrain_model/best_xy.pth")
            self.load_param_self_backbone("/media/zeh/2TBlue/FPI/pretrain_model/best_1500.pth")

        else:
            self.model_uav.transformer.load_param(pretrain)
            # self.load_param_self_backbone("/media/zeh/2TBlue/FPI/pretrain_model/best_1500.pth")


    def load_param_self_backbone(self, pretrained=None):
        if isinstance(pretrained, str):
            model_dict = self.state_dict()
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict, strict=False)
            for name, value in self.named_parameters():
                if name.split(".")[0]=="model_uav":
                    value.requires_grad = False
                else:
                    print(name)
            # print(model_dict)

    # def load_param_self_backbone(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         model_dict = self.model_uav.transformer.state_dict()
    #         model_dict2 = self.state_dict()
    #         state_dict = torch.load(pretrained)
    #         state_dict2 = {k: v for k, v in state_dict.items() if (k in model_dict)}
    #         self.model_uav.transformer.load_state_dict(state_dict2, strict=False)
    #         # 载入预训练模型参数后...
    #         for name, value in self.model_uav.transformer.named_parameters():
    #             if name in state_dict:
    #                 value.requires_grad = False
    #             else:
    #                 print(name)





class SiamUAV_CNN_Model(nn.Module):
    def __init__(self, opt):
        super(SiamUAV_CNN_Model, self).__init__()
        backbone = opt.backbone
        self.model_uav = make_cnn_model(backbone)
        if not opt.share:
            self.model_satellite = make_cnn_model(backbone)
        self.head = SiamFC_HEAD()
        self.opt = opt

    def forward(self, z, x):
        z = self.model_uav(z)
        if self.opt.share:
            x = self.model_uav(x)
        else:
            x = self.model_satellite(x)
        map = self.head(z, x)
        return map, None


def make_model(opt, pretrain=False):
    if opt.backbone in Transformer_model_list:
        model = SiamUAV_Transformer_Model(opt)
        if pretrain:
            model.load_params(opt)
        # if opt.backbone == "Swin-Transformer-S":
        #     model.model_uav.transformer.norm = nn.LayerNorm(384)
        #     model.model_satellite.transformer.norm = nn.LayerNorm(384)
    if opt.backbone in CNN_model_list:
        model = SiamUAV_CNN_Model(opt)
    return model

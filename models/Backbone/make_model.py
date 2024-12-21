import torch
import torch.nn as nn
from .backbones.vit_pytorch import vit_small_patch16_224_FSRA, deit_small_patch16_224_FSRA
import torch.nn.functional as F
from .backbones.swin_transformer import SwinTransformer
from resnest.torch import resnest50,resnest101
from torch.nn import AvgPool2d
from torchvision import models
from .backbones.os_pcpvt import os_pcpvt_small,os_pcpvt_base
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class build_CNN(nn.Module):
    def __init__(self, backbone_name = "Resnet50",pretrained=True):
        super(build_CNN, self).__init__()
        if backbone_name == "Resnet50":
            model = models.resnet50(pretrained=pretrained)
            # last stride = 1
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].conv2.stride = (1, 1)
        elif backbone_name == "Resnest50":
            model = resnest50(pretrained=pretrained)
            # last stride = 1
            model.layer4[0].downsample[0].stride = (1, 1)
            model.layer4[0].avd_layer.stride = 1
            model.layer4[0].downsample[0] = AvgPool2d(kernel_size=3, stride=(1, 1), padding=1)
        else:
            raise NameError("the backbone name is not exist!!!")

        self.model = model



    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        c1 = self.model.maxpool(x)
        c2 = self.model.layer1(c1)
        c3 = self.model.layer2(c2)
        c4 = self.model.layer3(c3)
        c5 = self.model.layer4(c4)
        return c5


class build_transformer(nn.Module):
    def __init__(self, opt,size, transformer_name="Vit-S"):
        super(build_transformer, self).__init__()

        print('using Transformer_type: {} as a backbone'.format(transformer_name))

        if transformer_name == "Vit-S":
            self.transformer = vit_small_patch16_224_FSRA(z_size=opt.UAVhw,
                                                           x_size=opt.Satellitehw,
                                                          stride_size=[16, 16],
                                                          )
        elif transformer_name == "Deit-S":
            self.transformer = deit_small_patch16_224_FSRA(z_size=opt.UAVhw,
                                                           x_size=opt.Satellitehw,
                                                           stride_size=[16, 16],
                                                           )
        elif transformer_name == "Swin-Transformer-S":
            self.transformer = SwinTransformer(img_size=size[0],
                                               embed_dim=96,
                                               depths=[2, 2, 18, 2],
                                               num_heads=[3, 6, 12, 24],
                                               drop_path_rate=0.3)
        elif transformer_name == "os_pcpvt_small":
            self.transformer = os_pcpvt_small()


    def forward(self, x,z):
        features = self.transformer(x,z)
        return features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_transformer_model(opt,size, transformer_name):
    model = build_transformer(opt,size, transformer_name=transformer_name)
    return model


def make_cnn_model(backbone_name,pretrained=True):
    model = build_CNN(backbone_name=backbone_name,pretrained=pretrained)
    return model




"""
-*- coding: utf-8 -*-

@作者(Author) : Chen Jiahao
@时间(Time) : 2023/4/30 19:51
@File : grad_cam_show.py
"""
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from models.model_stages_double import BiSeNet
from grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from models.model import make_model
import yaml
import argparse
from datasets.SiamUAV import SiamUAV_test
from tqdm import tqdm


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()

def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/media/zeh/4d723c17-52ed-4771-9ff5-c5b4cf1675e9/cjh/SiamUAV_date',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=16, type=int, help='')
    parser.add_argument('--checkpoint', default="../net_040.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--cuda', default=0, type=int, help='use panet or not')
    opt = parser.parse_args()
    config_path = '../opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    opt.UAVhw = config["UAVhw"]
    opt.Satellitehw = config["Satellitehw"]
    opt.share = config["share"]
    opt.backbone = config["backbone"]
    opt.padding = config["padding"]
    opt.centerR = config["centerR"]
    opt.neck = config["neck"]
    return opt

def create_model(opt):
    torch.cuda.set_device(opt.cuda)
    model = make_model(opt)
    state_dict = torch.load(opt.checkpoint)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model


def create_dataset(opt):
    dataset_test = SiamUAV_test(opt.test_data_dir, opt, mode="merge_test_700-1800_cr0.95_stride100")
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=opt.num_worker,
                                              pin_memory=True)
    return dataloaders


def test(model, dataloader, opt):
    for uav, satellite, X, Y, uav_path, sa_path in tqdm(dataloader):
        rgb_img = np.float32(satellite) / 255
        z = uav.cuda()
        x = satellite.cuda()
        response, loc_bias,out = model(z, x)
        # response = torch.sigmoid(response)
        # map = response[0].squeeze().cpu().detach().numpy()
        input_tensor = [z,x]
        # 推理
        # output = model(input_tensor)[0]
        output = response
        normalized_masks = torch.softmax(output, dim=1).cpu()

        # 自己的数据集的类别
        sem_classes = [
            '__background__', 'round', 'nok', 'headbroken', 'headdeep', 'shoulderbroken'
        ]

        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        round_category = sem_class_to_idx["__background__"]
        round_mask = torch.argmax(normalized_masks[0], dim=0).detach().cpu().numpy()
        round_mask_uint8 = 255 * np.uint8(round_mask == round_category)
        round_mask_float = np.float32(round_mask == round_category)

        # 自己要放CAM的位置
        target_layers = [model.model_uav.transformer.blocks[0]]
        targets = [SemanticSegmentationTarget(round_category, round_mask_float)]

        with GradCAM(model=model, target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 保存CAM的结果
        img = Image.fromarray(cam_image)
        # img.show()
        img.save('./result.png')





# 推理结果图与原图拼接
# both_images = np.hstack((image, np.repeat(round_mask_uint8[:, :, None], 3, axis=-1)))
# img = Image.fromarray(both_images)
# img.save("./hhhh.png")








def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()
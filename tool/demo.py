# -*- coding: utf-8 -*-

from __future__ import print_function, division
import yaml
import warnings
from models.model import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
from datasets.SiamUAV import SiamUAV_test
warnings.filterwarnings("ignore")


def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/media/zeh/4d723c17-52ed-4771-9ff5-c5b4cf1675e9/cjh/SiamUAV_date', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=4, type=int, help='')
    parser.add_argument('--checkpoint', default="net_018.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--filterR', default=7, type=int, help='')
    parser.add_argument('--fpn', default=0, type=float, help='use fpn or not')
    parser.add_argument('--panet', default=1, type=float, help='use panet or not')
    parser.add_argument('--cuda', default=0, type=int, help='use panet or not')
    opt = parser.parse_args()
    config_path = 'opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream,Loader=yaml.FullLoader)
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
    pretrained_dict = {k.split("module.")[-1]: v for k, v in state_dict.items()}
    model.load_state_dict(pretrained_dict)
    model = model.cuda()
    model.eval()
    return model

def create_dataset(opt):
    dataset_test = SiamUAV_test(opt.test_data_dir, opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=opt.num_worker,
                                              pin_memory=True)
    return dataloaders


def evaluate(opt, pred_XY, label_XY):
    pred_X,pred_Y = pred_XY
    label_X,label_Y = label_XY
    x_rate = (pred_X-label_X)/opt.Satellitehw[0]
    y_rate = (pred_Y-label_Y)/opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate)+np.square(y_rate))/2) # take the distance to the 0-1
    result = np.exp(-1*opt.k*distance)
    return result


def test(model,dataloader,opt):
    for uav,satellite,X,Y,UAV_path,Satellite_path in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response = model(z,x)
        map1 = response.squeeze().cpu().detach().numpy()

        # h, w = map.shape
        satellite_map1 = cv2.resize(map1,opt.Satellitehw)


        id1 = np.argmax(satellite_map1)


        S_X1 = int(id1//opt.Satellitehw[0])
        S_Y1 = int(id1%opt.Satellitehw[1])


        pred_XY1 = np.array([S_X1,S_Y1])

        label_XY = np.array([X.squeeze().detach().numpy(),Y.squeeze().detach().numpy()])
        uavImage = cv2.imread(UAV_path[0])
        satelliteImage = cv2.imread(Satellite_path[0])
        uavImage = cv2.resize(uavImage,opt.UAVhw)
        satelliteImage = cv2.resize(satelliteImage,opt.Satellitehw)

        satelliteImage = cv2.circle(satelliteImage,pred_XY1[::-1].astype(int),radius=5,color=(255,0,0),thickness=3)

        satelliteImage = cv2.circle(satelliteImage,label_XY[::-1].astype(int),radius=5,color=(0,255,0),thickness=3)
        cv2.imshow("result",satelliteImage)
        cv2.imshow("uav",uavImage)
        cv2.waitKey(0)




def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model,dataloader,opt)

if __name__ == '__main__':
    main()



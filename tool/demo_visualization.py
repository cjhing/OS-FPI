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
from tool.JWD_M import Distance
import json
import os

warnings.filterwarnings("ignore")

# visualize the localization result and heat map drawing
def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/media/###/SiamUAV_date',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--checkpoint', default="../net_040.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--cuda', default=3, type=int, help='use panet or not')
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


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.Satellitehw[0]
    y_rate = (pred_Y - label_Y) / opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * opt.k * distance)
    return result

def create_hanning_mask(center_R):
    hann_window = np.outer(
        np.hanning(center_R + 2),
        np.hanning(center_R + 2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def heatmap(map,satelliteImage,centerR,Satellitehw,X,Y,sa_path):
    map = torch.sigmoid(map)
    map = map.squeeze().cpu().detach().numpy()
    map = cv2.resize(map, Satellitehw)
    kernel = create_hanning_mask(centerR)
    map = cv2.filter2D(map, -1, kernel)

    response_copy = map

    id1 = np.argmax(response_copy)
    S_X = int(id1 // Satellitehw[0])
    S_Y = int(id1 % Satellitehw[1])


    get_gps_x = S_X / Satellitehw[0]
    get_gps_y = S_Y / Satellitehw[0]
    path = sa_path[0].split("/")
    read_gps = json.load(
        open(sa_path[0].split("/Satellite")[0] + "/GPS_info.json", 'r', encoding="utf-8"))
    tl_E = read_gps["Satellite"][path[-1]]["tl_E"]
    tl_N = read_gps["Satellite"][path[-1]]["tl_N"]
    br_E = read_gps["Satellite"][path[-1]]["br_E"]
    br_N = read_gps["Satellite"][path[-1]]["br_N"]
    UAV_GPS_E = read_gps["UAV"]["E"]
    UAV_GPS_N = read_gps["UAV"]["N"]
    PRE_GPS_E = tl_E + (br_E - tl_E) * get_gps_y
    PRE_GPS_N = tl_N - (tl_N - br_N) * get_gps_x

    d_m = Distance(UAV_GPS_E, UAV_GPS_N, PRE_GPS_E, PRE_GPS_N)

    pred_XY1 = np.array([S_X, S_Y])

    label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])



    heatmap = normalization(response_copy)

    heatmap = np.uint8(255 * heatmap)  # 8bite

    heatmap = cv2.applyColorMap(heatmap, 2)
    satelliteImage = cv2.addWeighted(heatmap, 0.6, satelliteImage, 0.6, 0)

    satelliteImage = cv2.circle(satelliteImage, pred_XY1[::-1].astype(int), radius=2, color=(255, 0, 0), thickness=2)

    satelliteImage = cv2.circle(satelliteImage, label_XY[::-1].astype(int), radius=2, color=(0, 255, 0),
                                thickness=2)


    return satelliteImage

def test(model, dataloader, opt):
    i = 0
    for uav, satellite, X, Y, UAV_path, sa_path in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response, _ = model(z, x)

        uavImage = cv2.imread(UAV_path[0])
        satelliteImage = cv2.imread(sa_path[0])
        uavImage = cv2.resize(uavImage, opt.UAVhw)
        q1 = cv2.resize(satelliteImage, opt.Satellitehw)


        satelliteImage =heatmap(response[0],q1,opt.centerR,opt.Satellitehw,X,Y,sa_path)

        img =satelliteImage

        Newdir = os.path.join("/media/####/", str(i).zfill(5) + ".jpg")
        cv2.imwrite(Newdir, img)
        i = i + 1


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()

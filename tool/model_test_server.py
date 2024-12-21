"""
-*- coding: utf-8 -*-

@作者(Author) : Chen Jiahao
@时间(Time) : 2023/3/15 10:08
@File : model_test_server.py
"""
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import json
import time
from torch.nn.functional import sigmoid
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

warnings.filterwarnings("ignore")

map_all = [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
save_metre_rnage = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
save_metre_original = (np.array([0] * (len(save_metre_rnage) + 1) * len(map_all)).reshape(len(map_all),
                                                                                len(save_metre_rnage) + 1)).tolist()
save_metre_xy = (np.array([0] * (len(save_metre_rnage) + 1) * len(map_all)).reshape(len(map_all),
                                                                                len(save_metre_rnage) + 1)).tolist()
d_m_all=0
save_metre_original_all = save_metre_original[0].copy()
save_metre_xy_all = save_metre_xy[0].copy()
def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='D:/python_project/FPI_DEV/DATASAT',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=2, type=int, help='')
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


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R + 2),
        np.hanning(center_R + 2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


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


def evaluate_distance(X, Y, opt, sa_path,bias=False):
    global map_all
    global save_metre_original
    global save_metre_xy
    global save_metre_rnage
    global d_m_all
    #################获取预测的经纬度信息#############################
    get_gps_x = X / opt.Satellitehw[0]
    get_gps_y = Y / opt.Satellitehw[0]
    path = sa_path[0].split("\\")
    read_gps = json.load(
        open(sa_path[0].split("\\Satellite")[0] + "/GPS_info.json", 'r', encoding="utf-8"))
    tl_E = read_gps["Satellite"][path[-1]]["tl_E"]
    tl_N = read_gps["Satellite"][path[-1]]["tl_N"]
    br_E = read_gps["Satellite"][path[-1]]["br_E"]
    br_N = read_gps["Satellite"][path[-1]]["br_N"]
    map_size = int(read_gps["Satellite"][path[-1]]["map_size"])
    UAV_GPS_E = read_gps["UAV"]["E"]
    UAV_GPS_N = read_gps["UAV"]["N"]
    PRE_GPS_E = tl_E + (br_E - tl_E) * get_gps_y  # 经度
    PRE_GPS_N = tl_N - (tl_N - br_N) * get_gps_x  # 纬度
    #################获取预测的经纬度信息#############################
    d_m = Distance(UAV_GPS_N, UAV_GPS_E, PRE_GPS_N, PRE_GPS_E)
    map_index = map_all.index(map_size)
    if bias==False:
        save_metre_original[map_index][21] = save_metre_original[map_index][21] + 1
        save_metre_original_all[21] = save_metre_original_all[21] + 1
        for i in range(len(save_metre_rnage)):
            if d_m <= save_metre_rnage[i]:
                save_metre_original_all[i] = save_metre_original_all[i] + 1
                save_metre_original[map_index][i] = save_metre_original[map_index][i] + 1
        json.dump(save_metre_original, open("out.json", "w"), indent=2)
        d_m_all = d_m_all + 1
        if d_m_all % 1000 == 0:
            print('3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
                save_metre_original_all[0] / save_metre_original_all[21],
                save_metre_original_all[1] / save_metre_original_all[21],
                save_metre_original_all[2] / save_metre_original_all[21],
                save_metre_original_all[4] / save_metre_original_all[21],
                save_metre_original_all[6] / save_metre_original_all[21],
                save_metre_original_all[8] / save_metre_original_all[21],
                save_metre_original_all[10] / save_metre_original_all[21]))
            for i in range(len(map_all)):
                print('{}:3m:{}   5m:{}  10m:{}  20m:{}   30m:{}  40m:{}  50m:{}  '.format(map_all[i],
                                                                                           save_metre_original[i][0] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][1] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][2] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][4] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][6] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][8] / save_metre_original[i][
                                                                                               21],
                                                                                           save_metre_original[i][10] /
                                                                                           save_metre_original[i][21]))
    else:
        save_metre_xy[map_index][21] = save_metre_xy[map_index][21] + 1
        save_metre_xy_all[21] = save_metre_xy_all[21] + 1
        for i in range(len(save_metre_rnage)):
            if d_m <= save_metre_rnage[i]:
                save_metre_xy_all[i] = save_metre_xy_all[i] + 1
                save_metre_xy[map_index][i] = save_metre_xy[map_index][i] + 1
        json.dump(save_metre_xy, open("out_xy.json", "w"), indent=2)
        if d_m_all % 1000 == 0:
            print('3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
                save_metre_xy_all[0] / save_metre_xy_all[21],
                save_metre_xy_all[1] / save_metre_xy_all[21],
                save_metre_xy_all[2] / save_metre_xy_all[21],
                save_metre_xy_all[4] / save_metre_xy_all[21],
                save_metre_xy_all[6] / save_metre_xy_all[21],
                save_metre_xy_all[8] / save_metre_xy_all[21],
                save_metre_xy_all[10] / save_metre_xy_all[21]))
            for i in range(len(map_all)):
                print('{}:3m:{}   5m:{}  10m:{}  20m:{}   30m:{}  40m:{}  50m:{}  '.format(map_all[i],
                                                                                           save_metre_xy[i][0] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][1] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][2] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][4] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][6] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][8] / save_metre_xy[i][
                                                                                               21],
                                                                                           save_metre_xy[i][10] /
                                                                                           save_metre_xy[i][21]))


    return 0


def test(model, dataloader, opt):
    total_score = 0.0  # original data test
    total_score_b = 0.0  # add xy model test
    flag_bias = 0
    start_time = time.time()



    for uav, satellite, X, Y, uav_path, sa_path in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response, loc_bias = model(z, x)
        response = torch.sigmoid(response)
        map = response[0].squeeze().cpu().detach().numpy()

        if opt.centerR != 1:
            kernel = create_hanning_mask(opt.centerR)
            map = cv2.filter2D(map, -1, kernel)

        label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])

        satellite_map = cv2.resize(map, opt.Satellitehw)
        id = np.argmax(satellite_map)
        S_X = int(id // opt.Satellitehw[0])
        S_Y = int(id % opt.Satellitehw[1])

        pred_XY = np.array([S_X, S_Y])
        single_score = evaluate(opt, pred_XY=pred_XY, label_XY=label_XY)
        total_score += single_score
        #  ################################real_distance_caculate##################################  ##
        evaluate_distance(S_X, S_Y, opt, sa_path,bias=False)  #


        #  ################################real_distance_caculate##################################  ##
        if loc_bias is not None:
            flag_bias = 1
            loc = loc_bias.squeeze().cpu().detach().numpy()
            id_map = np.argmax(map)
            S_X_map = int(id_map // map.shape[-1])
            S_Y_map = int(id_map % map.shape[-1])

            pred_XY_map = np.array([S_X_map, S_Y_map])
            pred_XY_b = (pred_XY_map + loc[:, S_X_map, S_Y_map]) * opt.Satellitehw[0] / loc.shape[-1]  # add bias
            pred_XY_b = np.array(pred_XY_b)
            single_score_b = evaluate(opt, pred_XY=pred_XY_b, label_XY=label_XY)
            total_score_b += single_score_b
            evaluate_distance(pred_XY_b[0], pred_XY_b[1], opt, sa_path,bias=True)  #

    print('3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
        save_metre_xy_all[0] / save_metre_xy_all[21],
        save_metre_xy_all[1] / save_metre_xy_all[21],
        save_metre_xy_all[2] / save_metre_xy_all[21],
        save_metre_xy_all[4] / save_metre_xy_all[21],
        save_metre_xy_all[6] / save_metre_xy_all[21],
        save_metre_xy_all[8] / save_metre_xy_all[21],
        save_metre_xy_all[10] / save_metre_xy_all[21]))
    print('3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
        save_metre_original_all[0] / save_metre_original_all[21],
        save_metre_original_all[1] / save_metre_original_all[21],
        save_metre_original_all[2] / save_metre_original_all[21],
        save_metre_original_all[4] / save_metre_original_all[21],
        save_metre_original_all[6] / save_metre_original_all[21],
        save_metre_original_all[8] / save_metre_original_all[21],
        save_metre_original_all[10] / save_metre_original_all[21]))
    time_consume = time.time() - start_time
    print("time consume is {}".format(time_consume))

    score = total_score / len(dataloader)
    print("the final score is {}".format(score))
    if flag_bias:
        score_b = total_score_b / len(dataloader)
        print("the final score_bias is {}".format(score_b))

    with open(opt.checkpoint + ".txt", "w") as F:
        F.write("the final score is {}\n".format(score))
        F.write("time consume is {}".format(time_consume))


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()

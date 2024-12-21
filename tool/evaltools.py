import numpy as np


def evaluate(opt, pred_XY, label_XY):
    opt.k=10
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.Satellitehw[0]
    y_rate = (pred_Y - label_Y) / opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * opt.k * distance)
    return result
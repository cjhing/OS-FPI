from __future__ import absolute_import, print_function

import os
import numpy as np
from torch.utils.data import Dataset
import glob
import json
from PIL import Image
import cv2
from .Augmentation import RandomCrop,RandomRotate,EdgePadding,RandomResize,RotateAndCrop
from torchvision import transforms
from .random_erasing import RandomErasing

class SiamUAV_test(Dataset):
    def __init__(self, root_dir, opt,mode="val"):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAV_test, self).__init__()
        self.root_dir = root_dir
        self.opt = opt
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir,mode)
        self.seq = glob.glob(os.path.join(self.root_dir_train,"*"))
        self.list_all_info = self.get_total_info()

    def get_total_info(self):
        list_all_info = []
        for seq in self.seq:
            UAV = os.path.join(seq,"UAV/0.JPG")
            Satellite_list = glob.glob(os.path.join(seq,"Satellite/*"))
            with open(os.path.join(seq, "labels.json"), 'r', encoding='utf8') as fp:
                json_context = json.load(fp)
            for s in Satellite_list:
                single_dict = {}
                single_dict["UAV"] = UAV
                single_dict["Satellite"] = s
                name = os.path.basename(s)
                single_dict["position"] = json_context[name]
                list_all_info.append(single_dict)
        return list_all_info


    def get_transformer(self):
        transform_uav_list = [
            transforms.Resize(self.opt.UAVhw, interpolation=3),
            transforms.ToTensor()
        ]

        transform_satellite_list = [
            transforms.Resize(self.opt.Satellitehw, interpolation=3),
            transforms.ToTensor()
        ]

        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.list_all_info)

    def __getitem__(self, index):
        single_info = self.list_all_info[index]
        UAV_image_path = single_info["UAV"]
        UAV_image = Image.open(UAV_image_path)
        UAV_image = self.transform["UAV"](UAV_image)

        Satellite_image_path = single_info["Satellite"]
        Satellite_image_ = Image.open(Satellite_image_path)
        Satellite_image = self.transform["satellite"](Satellite_image_)
        X,Y = single_info["position"]
        X = int(X/Satellite_image_.height*self.opt.Satellitehw[0])
        Y = int(Y/Satellite_image_.width*self.opt.Satellitehw[1])
        return [UAV_image,Satellite_image,X,Y,UAV_image_path,Satellite_image_path]


class SiamUAVCenter(Dataset):
    def __init__(self, root_dir, opt):
        '''
        :param root_dir: root of SiamUAV
        :param transform: a dict, format as {"UAV":Compose(),"Satellite":Compose()}
        '''
        super(SiamUAVCenter, self).__init__()
        self.opt = opt
        self.root_dir = root_dir
        self.transform = self.get_transformer()
        self.root_dir_train = os.path.join(self.root_dir,"train_center")
        self.seq = glob.glob(os.path.join(self.root_dir_train,"*"))
        self.SatelliteAugmentation = RandomCrop(cover_rate=opt.cover_rate,map_size=(512,opt.cut_circle))


    def get_transformer(self):
        transform_uav_list = [
            RandomResize(self.opt.UAVhw),
            # transforms.RandomRotation((-180,180)),
            # transforms.RandomApply([transforms.GaussianBlur(21, 10)], p=0.5),
            # RandomRotate(),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.3),
        ]

        transform_satellite_list = [
            RandomResize(self.opt.Satellitehw),
            transforms.ToTensor(),
        ]

        if self.opt.padding:
            transform_uav_list = [EdgePadding(self.opt.padding)] + transform_uav_list


        data_transforms = {
            'UAV': transforms.Compose(transform_uav_list),
            'Satellite': transforms.Compose(transform_satellite_list)
        }

        return data_transforms

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # load the json context
        UAV_image_path = os.path.join(self.seq[index],"UAV","0.JPG")
        UAV_image = Image.open(UAV_image_path)
        # UAV_image = self.UAVAugmentation(UAV_image)
        # UAV_image.show()
        UAV_image = self.transform["UAV"](UAV_image)

        Satellite_image_path = os.path.join(self.seq[index],"Satellite","0.tif")
        Satellite_image = Image.open(Satellite_image_path)
        Satellite_image,[ratex,ratey] = self.SatelliteAugmentation(Satellite_image)
        Satellite_image = self.transform["Satellite"](Satellite_image)

        return [UAV_image,Satellite_image,ratex,ratey]

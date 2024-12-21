import numpy as np
from PIL import Image
import random
import cv2
from torchvision import transforms
import math


class RandomCrop(object):
    """
    random crop from satellite and return the changed label
    """
    def __init__(self, cover_rate=0.9,map_size=(512,1000)):
        """
        map_size: (low_size,high_size)
        cover_rate: the max cover rate
        """
        self.cover_rate = cover_rate
        self.map_size = map_size


    def __call__(self, img):
        map_size = np.random.randint(int(self.map_size[0]),int(self.map_size[1]))
        if map_size%2==1:
            map_size-=1
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h,w,c = img.shape
        cx,cy = h//2,w//2
        # bbox = np.array([cx-self.map_size//2,cy-self.map_size//2,cx+self.map_size//2,cy+self.map_size//2],dtype=np.int)
        # new_map = img[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1,:]
        # assert new_map.shape[0:2] == [self.map_size,self.map_size], "the size is not correct"
        RandomCenterX = np.random.randint(int(0.5*h-self.cover_rate/2*map_size),int(0.5*h+self.cover_rate/2*map_size))
        RandomCenterY = np.random.randint(int(0.5*w-self.cover_rate/2*map_size),int(0.5*w+self.cover_rate/2*map_size))
        bbox = np.array([RandomCenterX-map_size//2,
                         RandomCenterY-map_size//2,
                         RandomCenterX+map_size//2,
                         RandomCenterY+map_size//2],dtype=int)

        bias_left = bias_top = bias_right = bias_bottom = 0
        if bbox[0] < 0:
            bias_top = int(-1 * bbox[0])
            bbox[0] = 0
        if bbox[1] < 0:
            bias_left = int(-1 * bbox[1])
            bbox[1] = 0
        if bbox[2] >= w:
            bias_bottom = int(bbox[2] - h)
            bbox[2] = w
        if bbox[3] >= h:
            bias_right = int(bbox[3] - w)
            bbox[3] = h
        croped_image = img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3]), :]
        mean = np.mean(croped_image, axis=(0, 1), dtype=float)
        # padding the border
        croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right,
                                          cv2.BORDER_CONSTANT,
                                          value=mean)
        ratex = 0.5+(cx-RandomCenterX)/map_size
        ratey = 0.5+(cy-RandomCenterY)/map_size
        # cv2.imshow("sf",cv2.circle(croped_image,(int(ratey*map_size),int(ratex*map_size)),radius=5,color=(255,0,0),thickness=2))
        # cv2.waitKey(0)

        assert croped_image.shape[0:2] == (map_size, map_size), "the size is not correct the cropped size is {},the map_size is {}".format(croped_image.shape[:2],map_size)
        image = Image.fromarray(croped_image.astype('uint8')).convert('RGB')
        return image,[ratex,ratey]


class RandomRotate(object):
    """
    rotate 0 90 180 or 270 degree
    """
    def __init__(self):
        super(RandomRotate, self).__init__()
        self.random_list=[0,1,2,3]
        
    def __call__(self, img):
        random_rate = np.random.choice(self.random_list)
        image = img.rotate(int(random_rate*90))
        return image


class EdgePadding(object):
    def __init__(self,padding_times):
        super(EdgePadding, self).__init__()
        self.padding_times = padding_times


    def __call__(self, img):
        image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        h,w,c = image.shape
        pad_top = pad_bottom = h*self.padding_times//2
        pad_left = pad_right = w*self.padding_times//2
        image_pad = cv2.copyMakeBorder(image,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_CONSTANT,value=[0,0,0])
        img = Image.fromarray(cv2.cvtColor(image_pad, cv2.COLOR_BGR2RGB))
        return img


class RandomResize(object):
    def __init__(self,img_size):
        super(RandomResize, self).__init__()
        self.resize_list=[0,1,2,3]
        self.img_size = img_size

    def __call__(self, img):
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        resize_method = np.random.choice(self.resize_list)
        image = cv2.resize(image,self.img_size,interpolation=resize_method)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class RotateAndCrop(object):
    def __init__(self,rate=0.5):
        self.rate = rate

    def __call__(self, img):
        img_=np.array(img).copy()

        def getPosByAngle(img, angle):
            h, w, c = img.shape
            x_center = y_center = h // 2
            r = h // 2
            angle_lt = angle - 45
            angle_rt = angle + 45
            angle_lb = angle - 135
            angle_rb = angle + 135
            angleList = [angle_lt, angle_rt, angle_lb, angle_rb]
            pointsList = []
            for angle in angleList:
                x1 = x_center + r * math.cos(angle * math.pi / 180)
                y1 = y_center + r * math.sin(angle * math.pi / 180)
                pointsList.append([x1, y1])
            pointsOri = np.float32(pointsList)
            pointsListAfter = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
            M = cv2.getPerspectiveTransform(pointsOri, pointsListAfter)
            res = cv2.warpPerspective(img, M, (h, w))
            return res
        if np.random.random()>self.rate:
            image = img
        else:
            angle = int(np.random.random()*360)
            new_image = getPosByAngle(img_,angle)
            image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


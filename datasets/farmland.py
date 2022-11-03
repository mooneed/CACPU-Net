import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import imageio
import cv2
import matplotlib.pyplot as plt
import re
import random
import os

class FarmLand(Dataset):
    def __init__(self,
                data_path, 
                data,
                phase='train', 
                multi_scale=True, 
                scale_factor=16,
                flip=True,
                downsample_rate=1,
                ignore_label=4,
                base_size=256
                ):
        super(FarmLand, self).__init__()
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.base_size = 256
        self.crop_size = (base_size, base_size)
        self.multi_scale = multi_scale
        self.scale_factor = scale_factor
        self.flip = flip
        self.ignore_label = ignore_label
        self.downsample_rate = downsample_rate
        src_list = []
        lab_list = []
        name_list = []# 防止传入的dataname_list的数据类型不是np数组
        for image_name in data:
            src_list.append(os.path.join(data_path, "img", image_name))
            lab_list.append(os.path.join(data_path, "label", image_name))
            name_list.append(image_name)
        if self.phase=="train":
            data_list = list(zip(src_list,lab_list,name_list))
            random.shuffle(data_list)
            src_list,lab_list,name_list = zip(*data_list)
        self.src_list = np.array(src_list)
        self.lab_list = np.array(lab_list)
        self.name_list = np.array(name_list)
        
        self.label_mapping = {0: 0, 1: 1, 2: 2, 3: 3,
            4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label
        }

    """迭代：根据标签值映射表，调整标签值"""
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    """pad边界"""
    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)
        return pad_image

    """随机裁剪"""
    def rand_crop(self, image, label):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]

        return image, label

    """尺度变换"""
    def multi_scale_aug(self, image, label, rand_scale=1, rand_crop=True):
        scale_size = np.int(self.base_size * rand_scale + 0.5)
        #print("scale size: {}".format(scale_size))
        image = cv2.resize(image, (scale_size, scale_size),
                            interpolation=cv2.INTER_LINEAR) # 双线性插值放缩
        label = cv2.resize(label, (scale_size, scale_size),
                            interpolation=cv2.INTER_NEAREST)
        #print("image multi scale shape: {}".format(image.shape))
        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    """数据增强"""
    def gen_sample(self, image, label, multi_scale=True, is_flip=True):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label,
                                                rand_scale=rand_scale)

        #image = self.random_brightness(image) # 随机亮度
        #image = self.input_transform(image) # 标准化

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(
                label,
                None,
                fx=self.downsample_rate,
                fy=self.downsample_rate,
                interpolation=cv2.INTER_NEAREST
            )
        return image, label

    def __getitem__(self, item):
        # image
        image_name = self.name_list[item]
        image = imageio.imread(self.src_list[item]) * 0.0001
        image = np.array(image)
        label = imageio.imread(self.lab_list[item])
        label = self.convert_label(label)
        label = np.array(label)
        if self.phase=="train":
            image, label = self.gen_sample(
                image, label, self.multi_scale, self.flip
            )
        else:
            image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy()).type(torch.FloatTensor)
        label = torch.from_numpy(label.copy()).type(torch.FloatTensor)
        #plt.imshow(image[0,:,:].numpy())
        #plt.savefig(os.path.join('./res/enhance_image',"{}_image.jpg".format(image_name[:-4])))
        #plt.imshow(label.numpy())
        #plt.savefig(os.path.join('./res/enhance_image',"{}_label.jpg".format(image_name[:-4])))
        return image_name, image, label
    
    def __len__(self):
        return len(self.name_list)
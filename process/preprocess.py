import torch
import imageio
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tqdm import tqdm
import os


"""预处理原始哨兵遥感图像"""
"""
    给定多光谱图像，裁剪对应序号的图像块
    arr: 需要执行padding操作的数组
    block_size: 裁剪图像块尺寸
    block_id: 裁剪图像块序号
"""
def getImageBlock(arr, block_id, block_size):
    """
        r,c: 原始图像行高列宽
        br,bc: 裁剪图像块行高列宽
        cnum: 原始图像列容纳图像块数（无需知道行容纳图像块数）
        idr,idc: 当前裁剪图像块id所处的行列数
        idrstart,idrend: 当前裁剪图像块的行起止像素数
        idcstart,idcend: 当前裁剪图像块的列起止像素数
    """
    _, r, c = arr.shape
    br, bc = block_size
    cnum = c / bc
    tem = block_id
    idr = int(tem // cnum)
    idc = int(tem % cnum)
    idrstart = br * idr
    idrend = br * (idr + 1)
    idcstart = bc * idc
    idcend = bc * (idc + 1)
    img= arr[:, idrstart:idrend, idcstart:idcend]
    return img

"""
    根据图像块尺寸padding原始图像
    arr: 需要执行padding操作的数组
    block_size: 根据block的尺寸，判断遍历需要的padding尺寸
"""
def padding(arr, block_size):
    """
        r,c: 原始图像尺寸
        br,bc: block尺寸
        row,col：padding后图像尺寸
        pr,pc: padding边框尺寸
    """
    _, r, c = arr.shape
    br, bc = block_size
    print("ori image shape: {}, target block size: {}".format(arr.shape,block_size))
    if r % br == 0:
        row = r
    else:
        row = r + (br - r % br)
    if c % bc == 0:
        col = c
    else:
        col = c + (bc - c % bc)
    pr = row - r
    pc = col - c
    arr = np.pad(
        arr,
        ((0, 0), (0, pr), (0, pc)),# 0维不填充，1维（行）上不填/下填满，2维（列）左不填/右填满
        "constant"# 常量填充
    )
    print("padding image shape: {}".format(arr.shape))
    return arr

"""切割图像块"""
def cutImageBlocks(path,block_size=(256,256),step=256):
    image_path = os.path.join(path,"img.tif")
    label_path = os.path.join(path,"label.tif")
    mask_path = os.path.join(path,"mask.tif")
    mask = np.array(imageio.imread(mask_path))[np.newaxis,:,:]
    image = np.array(imageio.imread(image_path) * 0.0001)
    image_tmp = np.abs(image)
    image = (image + image_tmp) / 2.0
    image = image.transpose(2,0,1)# 维度换位
    """原图简单预处理"""
    label = np.array(imageio.imread(label_path))[np.newaxis,:,:]# np.newaxis增加一个新维度
    """标签简单预处理"""
    label = label * mask
    print("##########cutImageBlocks##########")
    print("image shape:{},label shape:{},mask shape:{}".format(image.shape,label.shape,mask.shape))
    image_padding = padding(image,block_size)
    label_padding = padding(label,block_size)
    mask_padding = padding(mask,block_size)
    _, r, c = image_padding.shape
    block_num = int((r/block_size[0])*(c/block_size[1]))
    image_blocks = []
    label_blocks = []
    mask_blocks = []
    for block_id in range(block_num):
        image_block = getImageBlock(image_padding,block_id,block_size)# 性能差，每次都要传输全部原始图像
        label_block = getImageBlock(label_padding,block_id,block_size)
        mask_block = getImageBlock(mask_padding,block_id,block_size)
        image_name = "{}.tif".format(block_id)
        image_blocks.append(image_block)
        label_blocks.append(label_block)
        mask_blocks.append(mask_block)
        #dropout_ratio = 0.30# 海亮师哥预处理的弃置率偏高，因为田间道路等被掩码处理了
    image_blocks = np.array(image_blocks)
    label_blocks = np.array(label_blocks)
    mask_blocks = np.array(mask_blocks)
    return image_blocks,label_blocks
    

""""""
""""""
""""""
if __name__ == "__main__":
    path = "./rs_ss/1/"
    cutImageBlocks(
        path=path
    )
import sys
import os
from process.preprocess import cutImageBlocks
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.append(r"/disk1/repository/FeedImg/feed/DeepLab_v3_plus/datasets")
from farmland import FarmLand as farmland
import models
from modules.measure import SegmentationMetric
from config import config, update_config
from tqdm import tqdm
import numpy as np
import gdal
import time
import matplotlib.pyplot as plt
import zipfile
import json
from utils import get_file_list, get_dir_list
np.set_printoptions(precision=4)# np.float保留4位小数

"""保存栅格数据tif"""
def save_astif(image,res_path,filename="full_image.tif"):
    res_path = os.path.join(res_path,filename)
    bands, r, c = image.shape
    datatype = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")
    datas = driver.Create(res_path, c, r, bands,datatype)
    for i in range(bands):
        datas.GetRasterBand(i + 1).WriteArray(image[i])
    del datas

"""推理"""
def inferance(pth_path,res_path,data_type,padding_image_size=(4352,5632),ori_image_size=(4280,5505),block_size=(256,256)):
    assert data_type in ["block","full_image"]# block读取本地图像块文件列表，full_image读取原始图像
    data_path = "./data/full_image"
    visual_images_path = os.path.join(res_path,"visual")
    if not os.path.exists(visual_images_path):
        os.makedirs(visual_images_path)
    class_num = 6# 有两类占比特别小未被移除
    device = 'cuda:{}'.format(3) if torch.cuda.is_available() else 'cpu'
    metric = SegmentationMetric(class_num)
    metric_per = SegmentationMetric(class_num)# 每张图片的评估
    res_log_dict = {}
    res_full_image = np.zeros(padding_image_size)# 最后保存时只取(:4280,:5505)
    cnum = padding_image_size[1] // 256# full每行容纳图像块列数

    """读取数据，生成dataloader"""
    if data_type == "full_image":
        #images,labels = cutImageBlocks("./rs_ss/1/")
        #image_labels = np.concatenate((images,labels),axis=1)
        #print("image_labels shape:{}".format(image_labels.shape))
        print("##########Dataloader##########")
        test_txt = os.path.join("./data/full_image","totallist.txt")
        test_list = np.loadtxt(test_txt, dtype=str, delimiter="\n")
        test_datatset = farmland(
            data_path, 
            test_list, 
            phase='test',
            multi_scale=False,
            flip=False
        )
        test_dataloader = DataLoader(test_datatset,batch_size=1,shuffle=False)
    else:
        test_list = np.loadtxt(test_txt, dtype=str, delimiter="\n")
        test_datatset = farmland(
            data_path, 
            test_list, 
            phase='test',
            multi_scale=False,
            flip=False
        )
        test_dataloader = DataLoader(test_datatset,batch_size=1,shuffle=False)
    """导入模型"""
    print("##########Net##########")
    net = eval('models.'+config.MODEL.NAME+'.get_model')(config)
    net.to(device=device)
    if os.path.exists(os.path.join(pth_path,"best.pth")):
        print("#######")
        print("#######")
        print(os.path.join(pth_path,"best.pth"))
        print("#######")
        print("#######")
        net.load_state_dict(torch.load(os.path.join(pth_path,"best.pth")))
    """开始推理"""
    print("##########inferance##########")
    value_dict = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0}
    start = time.time()
    net.eval()
    for block_id,(item_name, initial_image, semantic_image) in tqdm(enumerate(test_dataloader),
                                            total=len(test_dataloader),
                                            desc='test'):
        print(item_name)
        initial_image = initial_image.to(device)
        semantic_image = semantic_image.to(device)

        if config.MODEL.NAME == "OURS":
            result = net(initial_image)
            semantic_image_pred = result["fine"]
        else:
            semantic_image_pred = net(initial_image)
        if config.MODEL.NUM_OUTPUTS > 1:
            semantic_image_pred = semantic_image_pred[-1]
        semantic_image_pred = semantic_image_pred.detach()
        semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
        semantic_image_pred = semantic_image_pred.argmax(dim=0)
        
        """维度压缩"""
        initial_image = torch.squeeze(initial_image.cpu(),0)[0:1,:,].reshape(256,256)# 仅使用第一通道
        semantic_image = torch.squeeze(semantic_image.cpu(), 0)
        semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
        for value in range(6):
            value_dict[str(value)] += len(np.extract(semantic_image_pred==value,semantic_image_pred))
        """填充到res full image上"""
        block_i = block_id // cnum
        block_j = block_id % cnum
        br,bc = block_size
        res_full_image[block_i*bc:(block_i+1)*bc, block_j*br:(block_j+1)*br] = semantic_image_pred.numpy()
        metric.addBatch(semantic_image_pred, semantic_image)
        metric_per.addBatch(semantic_image_pred, semantic_image)
        acc_per = metric_per.get_accuracy()
        type_acc_per = metric_per.get_accuracy_list()
        mIoU_per = metric_per.get_mIoU()
        IoU_per = metric_per.get_IoU_list()
        metric_per.reset()
        """单张图片结果保存"""
        fig = plt.figure(figsize=(10, 10))
        fontsize = 20
        ax = fig.add_subplot(2, 2, 1)# 真值图像
        ax.set_title("ground truth",fontsize=fontsize)
        ax.imshow(semantic_image)
        plt.axis('off')
        ax = fig.add_subplot(2, 2, 2)# 预测结果
        ax.set_title("predict image",fontsize=fontsize)
        ax_temp = ax.imshow(semantic_image_pred)
        plt.axis('off')
        ax = fig.add_subplot(2, 2, 3)# 原始图像
        ax.set_title("initial image",fontsize=fontsize)
        ax.imshow(initial_image)
        plt.axis('off')
        ax = fig.add_subplot(2, 2, 4)# 颜色条与评价指标结果
        ax.set_title("mIoU={} acc={}".format(round(mIoU_per,4),round(acc_per,4)),fontsize=fontsize)
        for IoU_id,IoU_item in enumerate(IoU_per):# 保存每个种类的
            plt.text(0, 0.5-IoU_id*0.1, "IoU{}={} acc{}={}".format(IoU_id,round(IoU_item,4),IoU_id,round(type_acc_per[IoU_id],4)),fontdict=dict(fontsize=fontsize))
        cb = fig.colorbar(ax_temp, ax=ax)
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_ticks([0,1,2,3,4,5])
        plt.axis('off')
        plt.savefig(os.path.join(visual_images_path, "{}.png".format(item_name[0][:-4])))
        plt.close()

    print("#####################")
    print("#####################")
    print(value_dict)
    print("#####################")
    print("#####################")
    end = time.time()
    hours = int((end-start)/60/60)
    mins = int(((end-start)/60)%60)
    seconds = int((end-start)%60)
    print("train cost time: {}:{}:{}".format(hours,mins,seconds))

    """结果输出"""
    print("##########结果图像简单处理##########")
    res_full_image = res_full_image[np.newaxis,:ori_image_size[0],:ori_image_size[1]]
    res_full_image = res_full_image.astype(int)
    # 统计像素值个数
    value_dict = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0}
    for row in tqdm(res_full_image[0]):
        for item in row:
            if str(item) in value_dict:
                value_dict[str(item)] = value_dict[str(item)] + 1
            else:
                value_dict[str(item)] = 1
    print("#####################")
    print("#####################")
    print(value_dict)
    print("#####################")
    print("#####################")
    save_astif(res_full_image,res_path)
    acc = metric.get_accuracy()
    res_log_dict['test_best_acc'] = acc
    mIoU = metric.get_mIoU()
    res_log_dict['test_best_mIoU'] = mIoU
    type_acc = metric.get_accuracy_list()
    type_IoU = metric.get_IoU_list()
    print("acc: {}, mIoU: {}".format(round(acc,4), round(mIoU,4)))
    acc_str = ""
    for type_acc_id, type_acc_item in enumerate(type_acc):
        acc_str += "acc_{}:{} ".format(type_acc_id, type_acc_item)
        res_log_dict['test_type{}_acc'.format(type_acc_id)] = type_acc_item
    print(acc_str)
    IoU_str = ""
    for type_IoU_id, type_IoU_item in enumerate(type_IoU):
        IoU_str += "IoU_{}:{} ".format(type_IoU_id, round(type_IoU_item,4))
        res_log_dict['test_type{}_mIoU'.format(type_IoU_id)] = type_IoU_item
    print(IoU_str)
    json.dump(
        res_log_dict,
        open(os.path.join(res_path,"res_log.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )
    return_acc = [acc,type_acc.reshape(6,-1)]
    return_mIoU = [mIoU,type_IoU.reshape(6,-1)]
    return return_acc, return_mIoU

def zip_res_images(res_path):
    with zipfile.ZipFile(os.path.join(res_path,"full_image_res.zip"),mode="w") as f:
        for dir_name in get_dir_list(res_path):
            dir_path = os.path.join(res_path,dir_name)
            for file_name in get_file_list(dir_path):
                file_path = os.path.join(dir_path,file_name)
                f.write(file_path)

if __name__ == "__main__":
    pth_path = os.path.join("./res", "OURS", "farmland1_Dice_Adam_exp_PRafterECAbest_time2022-6-7_23h7m11s")
    res_path = os.path.join("./res", "full_image_res/OURS_2022_9_29")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    inferance(
        pth_path=pth_path,
        res_path=res_path,
        data_type="full_image"
    )
    #zip_res_images(res_path)
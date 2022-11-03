import sys
import os
sys.path.append(os.path.abspath(r".")) # 增加当前python指令路径到package检索路径列表中
#print(sys.path) # 查看python解释器import package检索路径列表
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from datasets import farmland
sys.path.append(r"/disk1/repository/FeedImg/feed/DeepLab_v3_plus/datasets")
from farmland import FarmLand as farmland
import models
from modules.loss_module import CrossEntropy, SoftDice, Combo
from modules.measure import SegmentationMetric
from modules.early_stopping import EarlyStopping
import time
from tqdm import tqdm
from config import config, update_config
import matplotlib.pyplot as plt
import zipfile
import json
from utils import get_file_list, get_dir_list, get_cur_time

import numpy as np
np.set_printoptions(precision=4)# np.float保留4位小数

import argparse
"""训练脚本参数获取与参数融合（/experiments/XXX/XXX.yaml与/lib/config/default.config）"""
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default=None,
                        type=str)
    parser.add_argument('--res_path',
                        help='experiment configure res_path',
                        default='.res/',
                        type=str)
    parser.add_argument('--gpu',
                        help='Modify config GPU',
                        default=None,
                        type=int)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg != None:
        update_config(config, args)# 通过yaml文件更新config默认参数
    print(config.LOSS.BALANCE_WEIGHTS)
    return args

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    #print("input shape: {}".format(input.shape))
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def train_val(
            train_dataloader, 
            val_dataloader, 
            begin_epoch, 
            end_epoch, 
            loss,
            optimizer, 
            metric):
    #####################################
    # 训练
    # 1.训练
    # 2.验证
    # 3.验证结果展示
    # 4.训练时间记录与训练结果保存
    #####################################
    """开始训练"""
    start = time.time()# 开始训练计时
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=config.OPTIMIZER.LR * 0.01, last_epoch=-1
    )# 学习率调整
    """开始每轮训练"""
    best_id = -101
    train_loss = []
    val_loss = []
    train_acc = []
    train_mIoU = []
    val_acc = []
    val_mIoU = []
    train_type_acc = np.zeros(num_classes).reshape(num_classes,-1)
    val_type_acc = np.zeros(num_classes).reshape(num_classes,-1)
    for cur_epoch in range(begin_epoch + 1, end_epoch + 1):
        """构建训练可视化进度条"""
        train_bar = tqdm(train_dataloader,
            total=len(train_dataloader),# 仅定长数值可以可视化进度条，其他的则是可视化文本内容
            desc="train, cur_epoch:{0}".format(cur_epoch)# 可视化进度条头部文字
        )
        epoch_train_loss = 0
        epoch_val_loss = 0
        """开始每batch训练"""
        net.train()
        for (_, initial_image, semantic_image) in train_bar:
            initial_image = initial_image.to(device)
            semantic_image = semantic_image.to(device)

            if config.MODEL.NAME == "OURS" or config.MODEL.NAME == "UNet_CP" or config.MODEL.NAME == "UNet_ECA_CP" or config.MODEL.NAME == "UNet_PReLU_ECA_bCP":
                result = net(initial_image)
                semantic_image_pred = result["pred"]
                cur_seg_loss = loss(semantic_image_pred, semantic_image.long())
                gt_points = point_sample(
                    semantic_image.float().unsqueeze(1),
                    result["points"],
                    mode="nearest",
                    align_corners=False
                ).squeeze_(1)
                #print("point_pred shape: {}".format(result["point_pred"].shape))
                #print("gt_points shape: {}".format(gt_points.shape))
                cur_point_loss = point_loss(result["point_pred"],gt_points.long())
                cur_loss = cur_seg_loss + cur_point_loss
            else:
                semantic_image_pred = net(initial_image)
                cur_loss = loss(semantic_image_pred, semantic_image.long())# long()为向下取整
                if config.MODEL.NUM_OUTPUTS > 1:
                    semantic_image_pred = semantic_image_pred[-1]
            epoch_train_loss+=cur_loss.item()
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            """训练部分指标可视化"""
            visual_image_pred = semantic_image_pred.detach()
            visual_image_pred = F.softmax(visual_image_pred.squeeze(), dim=0)
            visual_image_pred = visual_image_pred.argmax(dim=1)

            visual_image = torch.squeeze(semantic_image.cpu(), 0)
            visual_image_pred = torch.squeeze(visual_image_pred.cpu(), 0)
            metric.addBatch(visual_image_pred, visual_image)
        lr_adjust.step()
        epoch_train_loss/=len(train_dataloader.dataset)
        train_loss.append(epoch_train_loss)
        acc = metric.get_accuracy()
        train_acc.append(acc)
        mIoU = metric.get_mIoU()
        train_mIoU.append(mIoU)
        type_acc = np.array(metric.get_accuracy_list()).reshape(num_classes,-1)
        train_type_acc = np.append(train_type_acc,type_acc,axis=1)
        metric.reset()

        """训练后即时验证"""
        with torch.no_grad():
            net.eval()
            for (_, initial_image, semantic_image) in tqdm(val_dataloader,
                                                    total=len(val_dataloader), 
                                                    desc='val'):
                initial_image = initial_image.to(device)
                semantic_image = semantic_image.to(device)

                if config.MODEL.NAME == "OURS" or config.MODEL.NAME == "UNet_CP" or config.MODEL.NAME == "UNet_ECA_CP" or config.MODEL.NAME == "UNet_PReLU_ECA_bCP":
                    result = net(initial_image)
                    semantic_image_pred = result["fine"]
                    cur_loss = loss(semantic_image_pred, semantic_image.long())
                else:
                    semantic_image_pred = net(initial_image)
                    cur_loss = loss(semantic_image_pred, semantic_image.long())# long()为向下取整
                epoch_val_loss+=cur_loss.item()
                if config.MODEL.NUM_OUTPUTS > 1:
                    semantic_image_pred = semantic_image_pred[-1]
                semantic_image_pred = F.softmax(semantic_image_pred.squeeze(), dim=0)
                semantic_image_pred = semantic_image_pred.argmax(dim=0)

                semantic_image = torch.squeeze(semantic_image.cpu(), 0)
                semantic_image_pred = torch.squeeze(semantic_image_pred.cpu(), 0)
                metric.addBatch(semantic_image_pred, semantic_image)
            epoch_val_loss/=len(val_dataloader.dataset)
            val_loss.append(epoch_val_loss)

        """验证部分评价指标可视化"""
        acc = metric.get_accuracy()
        val_acc.append(acc)
        mIoU = metric.get_mIoU()
        val_mIoU.append(mIoU)
        type_acc = np.array(metric.get_accuracy_list()).reshape(num_classes,-1)
        val_type_acc = np.append(val_type_acc,type_acc,axis=1)
        type_IoU = metric.get_IoU_list()
        print("acc: {}, mIoU: {}".format(round(acc,4), round(mIoU,4)))
        acc_str = ""
        for type_acc_id, type_acc_item in enumerate(type_acc):
            acc_str += "acc_{}:{} ".format(type_acc_id, round(type_acc_item[0],4))
        print(acc_str)
        IoU_str = ""
        for type_IoU_id, type_IoU_item in enumerate(type_IoU):
            IoU_str += "IoU_{}:{} ".format(type_IoU_id, round(type_IoU_item,4))
        print(IoU_str)
        metric.reset()
        """判断是否提前中断训练&保存"""
        update_model_flag = early_stopping(
            epoch_val_loss,
            net, 
            os.path.join(res_path,"best.pth")
        )
        if update_model_flag:
            best_id = cur_epoch - 1# 结果数组从0索引开始，epoch从1开始索引，故-1
        if early_stopping.early_stop:
            break
    
    """记录训练时间"""
    end = time.time()# 结束训练计时
    hours = int((end-start)/60/60)
    mins = int(((end-start)/60)%60)
    seconds = int((end-start)%60)
    print("train cost time: {}:{}:{}".format(hours,mins,seconds))
    res_log_dict['best_epoch'] = best_id + 1
    res_log_dict['val_best_acc'] = val_acc[best_id]# 添加val最佳准确率到待保存的json文件字典中
    res_log_dict['val_best_mIoU'] = val_mIoU[best_id]
    """保存训练与验证曲线图片"""
    fig = plt.figure(figsize=(20, 20))
    plt.rcParams.update({'font.size': 20})
    fontsize = 20
    ax = fig.add_subplot(2, 2, 1)# 损失曲线
    ax.set_title("Loss curve",fontsize=fontsize)
    ax.plot(range(1, len(val_loss) + 1), val_loss, label='val_loss')
    ax.plot(range(1, len(train_loss) + 1), train_loss, label='train_loss')
    ax.legend()
    ax = fig.add_subplot(2, 2, 2)# 准确率曲线
    ax.set_title("Accuracy curve",fontsize=fontsize)
    ax.plot(range(1, len(val_acc) + 1), val_acc, label='val_acc')
    ax.plot(range(1, len(train_acc) + 1), train_acc, label='train_acc')
    ax.legend()
    ax = fig.add_subplot(2, 2, 3)# 训练子准确率曲线
    ax.set_title("Sub train acc curve",fontsize=fontsize)
    train_type_acc = np.delete(train_type_acc,0,axis=1)# 删除新建np数组时的第一列0
    for acc_id, acc_item in enumerate(train_type_acc):
        for acc_item_i in range(len(acc_item)):
            if np.isnan(acc_item[acc_item_i]):
                acc_item[acc_item_i] = 0.
        """画出各种类的准确率曲线"""
        ax.plot(range(1, len(acc_item) + 1), acc_item, label='train_type{}_acc'.format(acc_id))
    ax.legend()
    ax = fig.add_subplot(2, 2, 4)# 验证子准确率曲线
    ax.set_title("Sub val acc curve",fontsize=fontsize)
    val_type_acc = np.delete(val_type_acc,0,axis=1)# 删除新建np数组时的第一列0
    for acc_id, acc_item in enumerate(val_type_acc):
        for acc_item_i in range(len(acc_item)):
            if np.isnan(acc_item[acc_item_i]):
                acc_item[acc_item_i] = 0.
        """画出各种类的准确率曲线"""
        ax.plot(range(1, len(acc_item) + 1), acc_item, label='val_type{}_acc'.format(acc_id))
    ax.legend()
    plt.savefig(os.path.join(curve_path,"training_curve.png"))
    plt.close()
    return_loss = [train_loss[best_id],val_loss[best_id]]
    return_acc = [train_acc[best_id],val_acc[best_id],train_type_acc[:,best_id].reshape(num_classes,-1),val_type_acc[:,best_id].reshape(num_classes,-1)]
    return_mIoU = [train_mIoU[best_id],val_mIoU[best_id]]
    return return_loss, return_acc, return_mIoU


def test_visual(test_dataloader, log_type, is_visual):
    #####################################
    # 测试&可视化&结果保存
    #####################################
    """测试"""
    if os.path.exists(os.path.join(res_path,"best.pth")):
        net.load_state_dict(torch.load(os.path.join(res_path,"best.pth")))

    start = time.time()
    net.eval()
    for (item_name, initial_image, semantic_image) in tqdm(test_dataloader,
                                            total=len(test_dataloader),
                                            desc=log_type):
        initial_image = initial_image.to(device)
        semantic_image = semantic_image.to(device)
        
        if config.MODEL.NAME == "OURS" or config.MODEL.NAME == "UNet_CP" or config.MODEL.NAME == "UNet_ECA_CP" or config.MODEL.NAME == "UNet_PReLU_ECA_bCP":
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

        metric.addBatch(semantic_image_pred, semantic_image)
        metric_per.addBatch(semantic_image_pred, semantic_image)
        acc_per = metric_per.get_accuracy()
        type_acc_per = metric_per.get_accuracy_list()
        mIoU_per = metric_per.get_mIoU()
        IoU_per = metric_per.get_IoU_list()
        metric_per.reset()
        """保存可视化结果图片"""
        if is_visual == True:
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
            image_type = "unknown"
            if item_name[0] in train_list:
                image_type = "train"
            elif item_name[0] in val_list:
                image_type = "val"
            elif item_name[0] in test_list:
                image_type = "test"
            plt.savefig(os.path.join(visual_images_path, "{}{}.png".format(
                image_type, item_name[0][:-4]
            )))
            plt.close()

    end = time.time()
    hours = int((end-start)/60/60)
    mins = int(((end-start)/60)%60)
    seconds = int((end-start)%60)
    print("{} cost time: {}:{}:{}".format(log_type,hours,mins,seconds))

    """结果输出"""
    acc = metric.get_accuracy()
    res_log_dict['{}_best_acc'.format(log_type)] = acc
    res_log_dict['{}_average_acc'.format(log_type)] = 0 # 初始化平均准确率
    mIoU = metric.get_mIoU()
    res_log_dict['{}_best_mIoU'.format(log_type)] = mIoU
    type_acc = metric.get_accuracy_list()
    type_IoU = metric.get_IoU_list()
    print("acc: {}, mIoU: {}".format(round(acc,4), round(mIoU,4)))
    acc_str = ""
    average_acc = 0.0
    for type_acc_id, type_acc_item in enumerate(type_acc):
        acc_str += "acc_{}:{} ".format(type_acc_id, round(type_acc_item,4))
        res_log_dict['{}_type{}_acc'.format(log_type, type_acc_id)] = type_acc_item
        average_acc+= type_acc_item
    average_acc = average_acc/4.0
    res_log_dict['{}_average_acc'.format(log_type)] = average_acc
    
    print(acc_str)
    IoU_str = ""
    for type_IoU_id, type_IoU_item in enumerate(type_IoU):
        IoU_str += "IoU_{}:{} ".format(type_IoU_id, round(type_IoU_item,4))
        res_log_dict['{}_type{}_mIoU'.format(log_type, type_IoU_id)] = type_IoU_item
    print(IoU_str)
    return_acc = [acc,type_acc.reshape(num_classes,-1)]
    return_mIoU = [mIoU,type_IoU.reshape(num_classes,-1)]
    return return_acc, return_mIoU

def zip_res_images():
    with zipfile.ZipFile(os.path.join(res_zip_path,"{}.zip".format(exp_name)),mode="w") as f:
        for dir_name in get_dir_list(res_path):
            dir_path = os.path.join(res_path,dir_name)
            for file_name in get_file_list(dir_path):
                file_path = os.path.join(dir_path,file_name)
                f.write(file_path)

if __name__ == '__main__':
    #####################################
    # 训练前准备
    # 1.超参数设置&模型导入
    # 2.路径初始化&检查文件保存路径
    # 3.损失函数，梯度下降优化器，评价指标类，提前中断训练类
    #####################################
    """超参数设置"""
    args = parse_args()
    exp_name = "{}{}_{}_exp_{}_time{}".format(
        config.DATASET.NAME,
        config.DATASET.ID,
        config.LOSS.NAME,
        config.EXP_ID,
        get_cur_time()
    )
    num_classes = config.DATASET.NUM_CLASSES
    cur_GPU = None
    if args.gpu != None:
        cur_GPU = args.gpu 
    else:
        cur_GPU = config.GPU
    device = 'cuda:{}'.format(cur_GPU) if torch.cuda.is_available() else 'cpu'
    num_GPU = 1
    """设置随机种子"""
    manual_seed = np.random.randint(1, 10000)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    cudnn.benchmark = True
    """路径初始化"""
    data_path = os.path.join(config.DATASET.ROOT, config.DATASET.ID)
    total_txt = os.path.join(data_path,"totallist.txt")
    train_txt = os.path.join(data_path,"trainlist.txt")
    val_txt = os.path.join(data_path,"vallist.txt")
    test_txt = os.path.join(data_path,"testlist.txt")
    res_path = os.path.join(args.res_path, exp_name)
    visual_images_path = os.path.join(res_path,"visual")
    curve_path = os.path.join(res_path,"curve")
    res_log_path = os.path.join(res_path,"log")
    res_zip_path = os.path.join('./res',"zip")
    """检查文件保存路径"""
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    if not os.path.exists(visual_images_path):
        os.makedirs(visual_images_path)
    if not os.path.exists(curve_path):
        os.makedirs(curve_path)
    if not os.path.exists(res_log_path):
        os.makedirs(res_log_path)
    if not os.path.exists(res_zip_path):
        os.makedirs(res_zip_path)
    """模型初始化"""
    net = eval('models.'+config.MODEL.NAME+'.get_model')(config)
    if config.MODEL.NAME == "OURS" or config.MODEL.NAME == "UNet_CP" or config.MODEL.NAME == "UNet_ECA_CP" or config.MODEL.NAME == "UNet_PReLU_ECA_bCP":
        point_loss = CrossEntropy(ignore_index=4,weight=torch.FloatTensor(config.LOSS.CLASSES_WEIGHT).to(device))
    try:
        net = eval('models.'+config.MODEL.NAME+'.get_model')(config)
    except Exception:
        raise ValueError("model name {} isn't among \
            'DeepLabv3_plus', 'UNet', 'MAResUNet', check the yaml file.".format(config.MODEL.NAME))
    """选择设备"""
    net.to(device=device)
    if num_GPU > 1:# 是否多线程导入数据？
        net = nn.DataParallel(net)
    """优化器初始化"""
    if config.OPTIMIZER.NAME == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config.OPTIMIZER.LR) # 0.0003
    elif config.OPTIMIZER.NAME == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), 
            lr=config.OPTIMIZER.LR, # 0.1
            momentum=config.OPTIMIZER.MOMENTUM, # 0.9
            weight_decay=config.OPTIMIZER.WEIGHT_DECAY # 0.0001
        )
    else:
        raise ValueError("optimizer must among 'Adam', 'SGD', check the yaml file.")
    """损失函数，评价指标类，提前中断训练类，结果json文件保存"""
    if config.LOSS.NAME == 'Cross_Entropy':
        loss = CrossEntropy(ignore_index=4,weight=torch.FloatTensor(config.LOSS.CLASSES_WEIGHT).to(device))
    elif config.LOSS.NAME == 'Dice':
        loss = SoftDice(num_classes=num_classes)
    elif config.LOSS.NAME == 'Combo':
        loss = Combo(
            num_classes=num_classes,
            ignore_index=4,
            weight=torch.FloatTensor(config.LOSS.CLASSES_WEIGHT).to(device)
        )
    else:
        raise ValueError("loss must among 'Cross_Entropy', 'Dice', 'Combo', check the yaml file.")
    metric = SegmentationMetric(num_classes)
    metric_per = SegmentationMetric(num_classes)# 每张图片的评估
    early_stopping = EarlyStopping(patience=20, verbose=True)# 几乎不做中断
    res_log_dict = {}

    """训练"""
    if config.TRAIN.TRAIN == True:
        """导入数据集"""
        train_list = np.loadtxt(train_txt, dtype=str, delimiter="\n")
        val_list = np.loadtxt(val_txt, dtype=str, delimiter="\n")
        if config.DATASET.DATASET_DIVIDED == False: # 若数据集未划分好
            """重新打乱数据集"""
            total_list = np.append(train_list, val_list)
            np.random.shuffle(total_list)# 随机打乱数据集
            train_list = total_list[:len(train_list)]
            val_list = total_list[len(train_list):]
            """保存当前数据集划分txt文件"""
            np.savetxt(os.path.join(res_path,"trainlist.txt"),train_list,fmt="%s",delimiter="\n")
            np.savetxt(os.path.join(res_path,"vallist.txt"),val_list,fmt="%s",delimiter="\n")
        """生成dataloader"""
        train_datatset = farmland(
            data_path, 
            train_list, 
            phase='train',
            multi_scale=config.TRAIN.MULTI_SCALE,
            flip=config.TRAIN.FLIP
        )
        val_datatset = farmland(
            data_path, 
            val_list, 
            phase='val'
        )
        train_dataloader = DataLoader(train_datatset, batch_size=config.TRAIN.BATCH_SIZE,shuffle=True)
        val_dataloader = DataLoader(val_datatset, batch_size=config.VAL.BATCH_SIZE,shuffle=True)
        """训练"""
        train_val(
            train_dataloader = train_dataloader, 
            val_dataloader = val_dataloader,
            begin_epoch = config.TRAIN.BEGIN_EPOCH,
            end_epoch = config.TRAIN.END_EPOCH,
            loss = loss,
            optimizer = optimizer,
            metric = metric
        )

    """测试与可视化"""
    test_list = np.loadtxt(test_txt, dtype=str, delimiter="\n")
    test_dataset = farmland(data_path, test_list, phase='test')
    test_dataloader = DataLoader(test_dataset,batch_size=config.TEST.BATCH_SIZE,shuffle=False)
    total_list = np.loadtxt(total_txt, dtype=str, delimiter="\n")
    total_dataset = farmland(data_path, total_list, phase='test')
    total_dataloader = DataLoader(total_dataset,batch_size=config.TEST.BATCH_SIZE,shuffle=False)
    test_visual(
        test_dataloader = test_dataloader,
        log_type = "test",
        is_visual = False
    )
    test_visual(
        test_dataloader = total_dataloader,
        log_type = "total",
        is_visual = config.VISUAL
    )

    """保存训练结果json 最佳准确率"""
    json.dump(
        res_log_dict,
        open(os.path.join(res_log_path,"res_log.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )

    """压缩实验结果文件目录，方便从工作区导出"""
    if config.ZIP == True:
        zip_res_images()
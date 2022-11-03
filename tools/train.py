import sys
import os
sys.path.append(os.path.abspath(r".")) # 增加当前python指令路径到package检索路径列表中
#print(sys.path) # 查看python解释器import package检索路径列表
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import farmland
from models.deeplabv3_plus import DeepLabv3_plus
from models.UNet import UNet
from models.MAResUNet import MAResUNet
from modules.measure import SegmentationMetric
from modules.early_stopping import EarlyStopping
from modules.k_fold import get_k_fold_data
from tqdm import tqdm
import matplotlib.pyplot as plt
import zipfile
import json
from utils import get_file_list, get_dir_list

import numpy as np
np.set_printoptions(precision=4)# np.float保留4位小数

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataid', type=str, default='1', help="'0' means old dataset, '1' means new dataset.")
parser.add_argument('--expid', type=str, default='0', help="all the same exp order id.")
parser.add_argument('-m', '--model', type=str, default='D', help="choose the training loss among 'D'='Deeplab v3+', 'U'='UNet', 'M'='MAResUNet'.")
parser.add_argument('-l', '--loss', type=str, default='CE_Dice', help="choose the training loss among 'Cross_Entropy', 'Dice', 'CE_Dice'.")
parser.add_argument('-l_1', '--loss_1_weight', type=int, default=1)
parser.add_argument('-l_2', '--loss_2_weight', type=int, default=1)
parser.add_argument('-o', '--optim', type=str, default='Adam', help="choose the training optim among 'Adam', 'SGD', 'Adam_SGD'.")
parser.add_argument('-e', '--epoch', type=int, default=100, help="100 is the epoch number of the exp used.")
parser.add_argument('-b', '--batchsize', type=int, default=16, help="16 is the batch size number of the exp used.")
parser.add_argument('-d', '--device', type=int, default=3, help="choose the training device.")
parser.add_argument('-k', '--k_fold', type=int, default=0, help="Choose whether to k_fold(k) or not(0).")
parser.add_argument('--train', type=int, default=1, help="Choose whether to train(1) or not(0).")
parser.add_argument('--enhance', type=int, default=0, help="Choose whether to enhance data(1) or not(0).")
parser.add_argument('--visual', type=int, default=1, help="Choose whether to visualize(1) or not(0).")
parser.add_argument('--zip', type=int, default=1, help="Choose whether to zip the result file(1) or not(0).")

"""
    Dice Loss
    Dice系数：语义分割的常用评价指标之一
"""
def diceCoeff(pred, gt, num_classes, smooth=1e-5, activation='sigmoid'):
    """ computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
    pred = activation_fn(pred).to(device)
    N = gt.size(0)
    pred_flat = pred.view(N, -1).to(device)
    gt_flat = gt.type(torch.FloatTensor).view(N, -1).to(device)
 
    intersection = (pred_flat * gt_flat).sum(1).to(device)
    unionset = pred_flat.sum(1) + gt_flat.sum(1).to(device)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'
 
    def __init__(self, num_classes, activation='sigmoid', reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
 
    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :].to(device), y_true == torch.full(y_true.size(), i * 1.0).to(device), num_classes=self.num_classes, activation=self.activation))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

def k_fold_train(data_path, total_list, epoch, loss_ce, loss_dice, optimizer, metric, k=10):
    train_loss_sum, val_loss_sum = 0., 0.
    train_acc_sum, val_acc_sum = 0., 0.
    train_mIoU_sum, val_mIoU_sum = 0., 0.
    # np数组，使用np函数求平均
    train_type_acc, val_type_acc = np.zeros(class_num).reshape(class_num,-1),np.zeros(class_num).reshape(class_num,-1)
    for i in range(1,k+1):
        """模型重置"""
        if args.model == 'D':
            net = DeepLabv3_plus(outclass=class_num, stride=8, pretrained=True)
        elif args.model == 'U':
            net = UNet(n_channels=3, n_classes=class_num)
        elif args.model == 'M':
            net = MAResUNet(num_channels=3,num_classes=class_num)
        else:
            raise ValueError("model must among 'D', 'U', 'M'.")
        """优化器重置"""
        if args.optim == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        elif args.optim == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=0.1,momentum=0.9,weight_decay=0.0001)
        else:
            raise ValueError("loss must among 'Adam', 'SGD'.")
        early_stopping.reset()#每次训练初始化
        train_datatset, val_datatset = get_k_fold_data(i-1,data_path,total_list,k=k)
        print("####################################")
        print("cur k fold id:{}".format(i))
        print("cur train dataset size:{}".format(train_datatset.__len__()))
        print("cur val dataset size:{}".format(val_datatset.__len__()))
        print("####################################")
        train_dataloader = DataLoader(train_datatset, batch_size=batch_size,shuffle=True)
        val_dataloader = DataLoader(val_datatset, batch_size=1,shuffle=True)
        loss, acc, mIoU = train_val(
            train_dataloader, 
            val_dataloader,
            epoch,
            loss_ce,
            loss_dice,
            optimizer,
            metric,
            k_fold_id=i
        )
        train_loss_sum+=loss[0]
        val_loss_sum+=loss[1]
        train_acc_sum+=acc[0]
        val_acc_sum+=acc[1]
        train_mIoU_sum+=mIoU[0]
        val_mIoU_sum+=mIoU[1]
        train_type_acc=np.append(train_type_acc,acc[2],axis=1)
        val_type_acc=np.append(val_type_acc,acc[3],axis=1)
        acc, mIoU = test_visual(is_visual=0, k_fold_id=i)
    k_fold_res_dict = {
        "train loss":train_loss_sum/10.,
        "val loss":val_loss_sum/10.,
        "train acc":train_acc_sum/10.,
        "val acc":val_acc_sum/10.,
        "train mIoU":train_mIoU_sum/10.,
        "val mIoU":val_mIoU_sum/10.,
    }
    train_type_acc = np.delete(train_type_acc,0,axis=1)
    val_type_acc = np.delete(val_type_acc,0,axis=1)
    for type_acc_id, type_acc_item in enumerate(np.nanmean(train_type_acc,axis=1)):
        k_fold_res_dict['train_type{}_acc'.format(type_acc_id)] = type_acc_item
    for type_acc_id, type_acc_item in enumerate(np.nanmean(val_type_acc,axis=1)):
        k_fold_res_dict['val_type{}_acc'.format(type_acc_id)] = type_acc_item
    json.dump(
        k_fold_res_dict,
        open(os.path.join(res_log_path,"k_fold_res_log.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )

def train_val(train_dataloader, val_dataloader, epoch, loss_ce, loss_dice, optimizer, metric, k_fold_id=None):
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
                    optimizer, 10, eta_min=learning_rate * 0.01, last_epoch=-1
                )# 学习率调整？
    """开始每轮训练"""
    best_id = None
    train_loss = []
    val_loss = []
    train_acc = []
    train_mIoU = []
    val_acc = []
    val_mIoU = []
    train_type_acc = np.zeros(class_num).reshape(class_num,-1)
    val_type_acc = np.zeros(class_num).reshape(class_num,-1)
    for cur_epoch in range(1, epoch + 1):
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

            semantic_image_pred = net(initial_image)

            cur_loss = l_1 * loss_ce(semantic_image_pred, semantic_image.long())+\
                l_2 * loss_dice(semantic_image_pred, semantic_image.long())# long()为向下取整
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
        type_acc = np.array(metric.get_accuracy_list()).reshape(class_num,-1)
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

                semantic_image_pred = net(initial_image).detach()# detach()为不计算梯度
                cur_loss = l_1 * loss_ce(semantic_image_pred, semantic_image.long())+\
                    l_2 * loss_dice(semantic_image_pred, semantic_image.long())# long()为向下取整
                epoch_val_loss+=cur_loss.item()
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
        type_acc = np.array(metric.get_accuracy_list()).reshape(class_num,-1)
        val_type_acc = np.append(val_type_acc,type_acc,axis=1)
        type_IoU = metric.get_IoU_list()
        print("acc: {}, mIoU: {}".format(round(acc,4), round(mIoU,4)))
        acc_str = ""
        for type_acc_id, type_acc_item in enumerate(type_acc):
            acc_str += "acc_{}:{} ".format(type_acc_id, type_acc_item[0])
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
            os.path.join(res_path,"best.pth" if k_fold_id==None else "best_{}.pth".format(k_fold_id))
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
    res_log_dict['val_best_acc' if k_fold_id==None else 'val_best_acc_{}'.format(k_fold_id)] = val_acc[best_id]# 添加val最佳准确率到待保存的json文件字典中
    res_log_dict['val_best_mIoU' if k_fold_id==None else 'val_best_mIoU_{}'.format(k_fold_id)] = val_mIoU[best_id]
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
    plt.savefig(os.path.join(curve_path,"training_curve.png" if k_fold_id==None else "training_curve_{}.png".format(k_fold_id)))
    plt.close()
    return_loss = [train_loss[best_id],val_loss[best_id]]
    return_acc = [train_acc[best_id],val_acc[best_id],train_type_acc[:,best_id].reshape(class_num,-1),val_type_acc[:,best_id].reshape(class_num,-1)]
    return_mIoU = [train_mIoU[best_id],val_mIoU[best_id]]
    return return_loss, return_acc, return_mIoU


def test_visual(is_visual, k_fold_id=None):
    #####################################
    # 测试&可视化&结果保存
    #####################################
    """测试"""
    test_list = np.loadtxt(test_txt, dtype=str, delimiter="\n")
    test_datatset = farmland(data_path, test_list, phase='test')
    test_dataloader = DataLoader(test_datatset,batch_size=1,shuffle=False)
    if os.path.exists(os.path.join(res_path,"best.pth" if k_fold_id==None else "best_{}.pth".format(k_fold_id))):
        net.load_state_dict(torch.load(os.path.join(res_path,"best.pth" if k_fold_id==None else "best_{}.pth".format(k_fold_id))))

    start = time.time()
    net.eval()
    for (item_name, initial_image, semantic_image) in tqdm(test_dataloader,
                                            total=len(test_dataloader),
                                            desc='test'):
        initial_image = initial_image.to(device)
        semantic_image = semantic_image.to(device)

        semantic_image_pred = net(initial_image).detach()
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
        if is_visual == 1:
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
            plt.savefig(os.path.join(visual_images_path, item_name[0]))
            plt.close()

    end = time.time()
    hours = int((end-start)/60/60)
    mins = int(((end-start)/60)%60)
    seconds = int((end-start)%60)
    print("train cost time: {}:{}:{}".format(hours,mins,seconds))

    """结果输出"""
    acc = metric.get_accuracy()
    res_log_dict['test_best_acc' if k_fold_id==None else 'test_best_acc_{}'.format(k_fold_id)] = acc
    mIoU = metric.get_mIoU()
    res_log_dict['test_best_mIoU' if k_fold_id==None else 'test_best_mIoU_{}'.format(k_fold_id)] = mIoU
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
    return_acc = [acc,type_acc.reshape(class_num,-1)]
    return_mIoU = [mIoU,type_IoU.reshape(class_num,-1)]
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
    args = parser.parse_args()
    exp_name = "{}_{}_{}_exp_{}".format(args.dataid,args.loss,args.optim,args.expid)
    batch_size = args.batchsize
    epoch = args.epoch
    class_num = 4 # 有两类占比特别小未被移除
    learning_rate = 0.0001 * 3
    beta1 = 0.5#
    cuda = True
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    num_workers = 1
    size_h = 256
    size_w = 256
    flip = 0
    band = 3
    num_GPU = 1
    """设置随机种子"""
    manual_seed = np.random.randint(1, 10000)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    cudnn.benchmark = True
    """模型导入"""
    if args.model == 'D':
        net = DeepLabv3_plus(outclass=class_num, stride=8, pretrained=True)
    elif args.model == 'U':
        net = UNet(n_channels=4, n_classes=class_num)
    elif args.model == 'M':
        net = MAResUNet(num_channels=3,num_classes=class_num)
    else:
        raise ValueError("model must among 'D', 'U', 'M'.")
    """选择设备"""
    net.to(device=device)
    if num_GPU > 1:# 是否多线程导入数据？
        net = nn.DataParallel(net)
    #torch.cuda.set_device(device)
    """路径初始化"""
    data_path = './data/{}'.format(args.dataid)
    total_txt = os.path.join(data_path,"totallist.txt")
    train_txt = os.path.join(data_path,"trainlist.txt")
    val_txt = os.path.join(data_path,"vallist.txt")
    test_txt = os.path.join(data_path,"testlist.txt")
    res_path = os.path.join('./res', net.name, exp_name)
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
    """损失函数，梯度下降优化器，评价指标类，提前中断训练类，结果json文件保存"""
    loss_ce = nn.CrossEntropyLoss(ignore_index=4,weight=torch.FloatTensor([0.2, 0.2, 0.4, 0.2]).to(device))
    loss_dice = SoftDiceLoss(num_classes=class_num)
    if args.loss == 'Cross_Entropy':
        l_1 = args.loss_1_weight
        l_2 = 0
    elif args.loss == 'Dice':
        l_1 = 0
        l_2 = args.loss_2_weight
    elif args.loss == 'CE_Dice':
        l_1 = args.loss_1_weight
        l_2 = args.loss_2_weight
    else:
        raise ValueError("loss must among 'Cross_Entropy', 'Dice', 'CE_Dice'.")
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1,momentum=0.9,weight_decay=0.0001)
    else:
        raise ValueError("loss must among 'Adam', 'SGD'.")
    metric = SegmentationMetric(class_num)
    metric_per = SegmentationMetric(class_num)# 每张图片的评估
    early_stopping = EarlyStopping(patience=50, verbose=True)# 几乎不做中断
    res_log_dict = {}

    """训练"""
    if args.train == 1 and args.k_fold == 0:
        """导入数据集"""
        train_list = np.loadtxt(train_txt, dtype=str, delimiter="\n")
        val_list = np.loadtxt(val_txt, dtype=str, delimiter="\n")
        """重新打乱数据集"""
        total_list = np.append(train_list, val_list)
        train_list, val_list = None, None
        fold_size = total_list.shape[0] // 9  # 每份的个数:数据总条数/折数（向下取整）
        np.random.shuffle(total_list)# 随机打乱数据集
        for j in range(9):
            idx = slice(j * fold_size, (j + 1) * fold_size)# slice(start,end,step)切片函数 得到测试集的索引
            cur_fold_list = total_list[idx]
            if j == 0:# 第0折作val
                val_list = cur_fold_list
            elif train_list is None:
                train_list = cur_fold_list
            else:
                train_list = np.append(train_list, cur_fold_list)
        """保存当前数据集划分txt文件"""
        np.savetxt(os.path.join(res_path,"trainlist.txt"),train_list,fmt="%s",delimiter="\n")
        np.savetxt(os.path.join(res_path,"vallist.txt"),val_list,fmt="%s",delimiter="\n")
        """生成dataloader"""
        train_datatset = farmland(data_path, train_list, phase='train', enhance_flag=args.enhance)
        val_datatset = farmland(data_path, val_list, phase='val', enhance_flag=args.enhance)
        train_dataloader = DataLoader(train_datatset, batch_size=batch_size,shuffle=True)
        val_dataloader = DataLoader(val_datatset, batch_size=1,shuffle=True)
        """训练"""
        train_val(
            train_dataloader, 
            val_dataloader,
            epoch,
            loss_ce,
            loss_dice,
            optimizer,
            metric
        )
        """测试与可视化"""
        test_visual(args.visual)
    elif args.train == 1 and args.k_fold != 0:
        train_list = np.loadtxt(train_txt, dtype=str, delimiter="\n")
        val_list = np.loadtxt(val_txt, dtype=str, delimiter="\n")
        total_list = np.append(train_list, val_list)
        k_fold_train(
            data_path=data_path,
            total_list=total_list,
            epoch=epoch,
            loss_ce=loss_ce,
            loss_dice=loss_dice,
            optimizer=optimizer,
            metric=metric,
            k=args.k_fold
        )
    else:
        """测试与可视化"""
        test_visual(args.visual)

    """保存训练结果json 最佳准确率"""
    json.dump(
        res_log_dict,
        open(os.path.join(res_log_path,"res_log.json"),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )

    if args.zip == 1:
        zip_res_images()
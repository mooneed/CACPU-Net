import numpy as np
import os
import json
from utils import get_dir_list, get_cur_time

#CONFIG = "HRNet_OCR.yaml"

loss_weight_list = [
    [1,1],
    [1,2],
    [1,5],
    [1,10],
    [2,1],
    [5,1],
    [10,1],
]

CONFIG_LIST = [

    #"DeepLabv3_plus.yaml",
    #"HRNet_OCR.yaml",
    #"HRNet.yaml",
    #"MAResUNet.yaml",

    #"UNet_SE.yaml",
    #"UNet_CA.yaml",
    #"UNet_CCA.yaml",
    #"UNet_CBAM.yaml",

    "UNet_Dice_PReLU_ECA_bCP.yaml",
    "UNet_Dice_PReLU_ECA_bCP.yaml",
    "UNet_Dice_PReLU_ECA_bCP.yaml"

    #"Loss_1_1.yaml",
    #"Loss_1_2.yaml",
    #"Loss_1_5.yaml",
    #"Loss_1_10.yaml",
    #"Loss_10_1.yaml",
    #"Loss_5_1.yaml",
    #"Loss_2_1.yaml"

    #"UNet.yaml",
    #"UNet_PReLU.yaml",
    #"UNet_Dice.yaml",
    #"UNet_ECA.yaml",
    #"UNet_CP.yaml",
    #"UNet_Dice_ECA.yaml",
    #"UNet_Dice_PReLU_ECA.yaml",
    #"UNet_Dice_ECA_CP.yaml",
    #"UNet_Dice_PReLU_ECA_CP.yaml"
]

if __name__ == "__main__":
    """多轮训练"""
    """for i in range(10):
        os.system("python ./tools/train_config_edtion.py")"""

    """不同模型实验"""
    GPU = 5
    for cur_exp in range(len(CONFIG_LIST)):
        """10折交叉验证，数据集划分"""
        k = 10
        image_num = 177
        data_path = "./data/K_FOLD"
        res_path = "./res/{}_time{}".format(CONFIG_LIST[cur_exp][:-5],get_cur_time())
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        total_txt = os.path.join(data_path,"totallist.txt")
        total_list = np.loadtxt(total_txt, dtype=str, delimiter="\n")
        for i in range(k):
            fold_size = image_num // k  # 每份的个数:数据总条数/折数（向下取整）
            np.random.shuffle(total_list)  # 随机打乱数据集
            train_list, val_list, test_list = None, None, None
            for j in range(k):
                idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数 得到测试集的索引
                cur_fold_list = total_list[idx]
                if j == i:  # 第i折作val
                    val_list = cur_fold_list
                elif j == (i+1)%k: # 第i+1折作test
                    test_list = cur_fold_list
                elif train_list is None:
                    train_list = cur_fold_list
                else:
                    train_list = np.append(train_list, cur_fold_list)
            # 末尾的6个样本划为训练集
            idx = slice(k * fold_size, image_num)
            cur_fold_list = total_list[idx]
            train_list = np.append(train_list, cur_fold_list)
            print("第{}折训练，训练集num={}，验证集num={}，测试集num={}。".format(i+1,len(train_list),len(val_list),len(test_list)))
            np.savetxt(os.path.join(data_path, "trainlist.txt"), train_list, fmt="%s", delimiter="\n")
            np.savetxt(os.path.join(data_path, "vallist.txt"), val_list, fmt="%s", delimiter="\n")
            np.savetxt(os.path.join(data_path, "testlist.txt"), test_list, fmt="%s", delimiter="\n")
            os.system("python ./tools/train_config_edtion.py --cfg={} --res_path={} --gpu={}".format(
                os.path.join("./experiments",CONFIG_LIST[cur_exp]),res_path,GPU))
            # 存放数据集划分文件
            #np.savetxt(os.path.join(res_path, "{}_trainlist.txt".format(i+1)), train_list, fmt="%s", delimiter="\n")
            #np.savetxt(os.path.join(res_path, "{}_vallist.txt".format(i+1)), val_list, fmt="%s", delimiter="\n")
            #np.savetxt(os.path.join(res_path, "{}_testlist.txt".format(i+1)), test_list, fmt="%s", delimiter="\n")
        """统计K_FOLD结果"""
        res_log_dict = {}
        best_OA = 0.0
        best_AA = 0.0
        best_mIoU = 0.0
        sum_OA = 0.0
        sum_AA = 0.0
        sum_mIoU = 0.0
        for dir_name in get_dir_list(res_path):
            log_name = os.path.join(res_path, dir_name,"log","res_log.json")
            json_dict = None
            with open(log_name, 'r') as f:
                json_dict = json.load(f)
            cur_OA = json_dict["test_best_acc"]
            cur_AA = json_dict["test_average_acc"]
            cur_mIoU = json_dict["test_best_mIoU"]
            if cur_OA > best_OA:#以OA为总判断条件
                best_OA = cur_OA
                best_AA = cur_AA
                best_mIoU = cur_mIoU
            sum_OA+=cur_OA
            sum_AA+=cur_AA
            sum_mIoU+=cur_mIoU
        res_log_dict['best_OA'] = best_OA
        res_log_dict['best_AA'] = best_AA
        res_log_dict['best_mIoU'] = best_mIoU
        res_log_dict['average_OA'] = sum_OA / 10.0
        res_log_dict['average_AA'] = sum_AA / 10.0
        res_log_dict['average_mIoU'] = sum_mIoU / 10.0
        json.dump(
            res_log_dict,
            open(os.path.join(res_path,"k_fold_res_log.json"),'w',encoding='utf-8'),
            indent=4,ensure_ascii=False
        )
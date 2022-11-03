import torch
from datasets import farmland
import numpy as np

"""返回第i折交叉验证时所需要的训练和测试数据集"""
def get_k_fold_data(i, data_path, total_list, k=10):
    assert k > 1
    fold_size = total_list.shape[0] // k  # 每份的个数:数据总条数/折数（向下取整）
    np.random.shuffle(total_list)# 随机打乱数据集
    train_list, val_list = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)# slice(start,end,step)切片函数 得到测试集的索引
        cur_fold_list = total_list[idx]
        if j == i:# 第i折作val
            val_list = cur_fold_list
        elif train_list is None:
            train_list = cur_fold_list
        else:
            train_list = np.append(train_list, cur_fold_list)
    return farmland(data_path, train_list), farmland(data_path, val_list)
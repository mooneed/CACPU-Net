from scipy.stats import wilcoxon
from utils import get_dir_list
import json
import os

"""
    获得实验结果
    value_type="test_best_acc" or "test_best_mIoU"
    return: list
"""
def get_exp_res(exp_name, value_type,
    root_path="./res", json_name="./log/res_log.json"):
    exp_res = []
    exp_path = os.path.join(root_path, exp_name)
    for exp_i in get_dir_list(exp_path):
        exp_i_json_path = os.path.join(exp_path, exp_i, json_name)
        exp_i_json = None
        with open(exp_i_json_path, 'r') as f:
            exp_i_json = json.load(f)
        exp_res.append(exp_i_json[value_type])
    return exp_res

def wilcoxon_test(value_type,json_res_type):
    res_log_dict = {}
    # 获取我的实验结果
    # 手动复制粘贴我的结果文件夹
    my_exp_name = "OURS"
    my_exp_res = get_exp_res(my_exp_name, value_type)
    # 获取全部竞争实验的结果
    exp_name_list = []
    for exp_name in get_dir_list("./res"):
        exp_name_list.append(exp_name)
    for exp_name in exp_name_list:
        if exp_name == my_exp_name:
            continue
        compet_exp_res = get_exp_res(exp_name, value_type)
        # 计算p值
        w = None
        p = None
        w,p = wilcoxon(
            x=my_exp_res, y=compet_exp_res, 
            zero_method='wilcox', correction=False,
            alternative='greater', mode='auto'
        )
        # 保存p值
        res_log_dict[exp_name] = [w,p]
    # 保存结果文件
    json.dump(
        res_log_dict,
        open(os.path.join("./res","wilcoxon_res_log_{}.json".format(json_res_type)),'w',encoding='utf-8'),
        indent=4,ensure_ascii=False
    )

if __name__ == "__main__":
    wilcoxon_test("test_best_acc","acc")
    wilcoxon_test("test_best_mIoU","mIoU")

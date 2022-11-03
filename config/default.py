import os

from yacs.config import CfgNode as CN

# _C为默认config对象，在训练时会和对应训练任务的yaml文件中的配置通过update_config融合/更新
_C = CN()

_C.EXP_ID = "Default"
_C.VISUAL = False # 是否生成可视化文件
_C.ZIP = False # 是否压缩实验结果文件目录
_C.AUTO_RESUME = False # 是否自动断点保存/续训

# 显卡相关配置参数
_C.GPU = 4
_C.GPUS = (4,) # 多卡参数

# 网络相关配置参数
_C.MODEL = CN()
_C.MODEL.NAME = 'OURS'
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.EXTRA = CN(new_allowed=True)

# HRNet_OCR网络OCR模块相关配置参数
_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

# 损失函数相关配置参数
_C.LOSS = CN()
_C.LOSS.NAME = 'Cross_Entropy'
_C.LOSS.CE_WEIGHT = 0.5
_C.LOSS.DICE_WEIGHT = 0.5
_C.LOSS.CLASSES_WEIGHT = [0.25, 0.25, 0.25, 0.25]
_C.LOSS.BALANCE_WEIGHTS = [1] # 多损失函数的权重平衡，如Combo损失函数交叉熵和Dice。（或平衡OCR的损失权重）

# 优化器相关配置参数
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'Adam'
_C.OPTIMIZER.LR = 0.0003
_C.OPTIMIZER.MOMENTUM = 0.9
_C.OPTIMIZER.WEIGHT_DECAY = 0.0001

# DATASET相关配置参数
# NAME 数据集（文件夹）名
# ROOT 数据集根路径
# ID 数据集批次
# DATASET_DIVIDED 数据集是否划分好：True则dataset类初始化读取train/val/testlist.txt    False则dataset类初始化读取totallist.txt
# NUM_CLASSES 数据集样本种类
# NUM_CHANNELS 数据集图像通道数
# IGNORE_LABEL 数据集在计算损失时需要忽略的标签值
_C.DATASET = CN()
_C.DATASET.NAME = 'farmland'
_C.DATASET.ROOT = './data'
_C.DATASET.ID = '1'
_C.DATASET.DATASET_DIVIDED = True
_C.DATASET.NUM_CLASSES = 4
_C.DATASET.NUM_CHANNELS = 4
_C.DATASET.IGNORE_LABEL = -1

# 训练相关配置参数
_C.TRAIN = CN()
_C.TRAIN.TRAIN = True # 是否训练
_C.TRAIN.RESUME = False
_C.TRAIN.SHUFFLE = True
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 150
_C.TRAIN.BATCH_SIZE = 16

_C.TRAIN.IMAGE_SIZE = [256, 256]  # width * height
_C.TRAIN.BASE_SIZE = 256
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = False
_C.TRAIN.MULTI_SCALE = False
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

# 验证相关配置参数
_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1

_C.VAL.IMAGE_SIZE = [256, 256]  # width * height
_C.VAL.BASE_SIZE = 256
_C.VAL.DOWNSAMPLERATE = 1

# 测试相关配置参数
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''
_C.TEST.BATCH_SIZE = 1

_C.TEST.IMAGE_SIZE = [256, 256]  # width * height
_C.TEST.BASE_SIZE = 256
_C.TEST.FLIP = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)


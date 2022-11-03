import os

model_list = ["D","M","U"]
loss_list = ["Cross_Entropy", "Dice", "CE_Dice"]
loss_weight_list = [[1,1]]
optim_list = ["Adam","SGD"]

def auto_train():
    for loss in loss_list:
        for loss_weight in loss_weight_list:
            for optim in optim_list:
                for expid in range(1,2):
                    print("开始训练：-l={} -l_1={} -l_2={} -o={} --expid={}".format(
                        loss,loss_weight[0],loss_weight[1],optim,expid
                    ))
                    """训练指令"""
                    os.system("python train.py -l={} -l_1={} -l_2={} -o={} --expid={}".format(
                        loss,loss_weight[0],loss_weight[1],optim,expid
                    ))

if __name__ == "__main__":
    #auto_train()
    os.system("python ./tools/train.py -m={} -e={} -l={} -l_1={} -l_2={} -o={} -k={} --expid={} --train={} --enhance={} --visual={} --zip={}".format(
        "U",# model
        100,# epoch
        "Cross_Entropy",# loss
        1,# Cross_Entropy weight
        1,# Dice weight
        "Adam",# optim
        0,# k_fold
        "classweight_balance25252525",# expid
        1,# whether train
        0,# whether enhance data
        1,# whether visual
        0 # whether zip the res files
    ))
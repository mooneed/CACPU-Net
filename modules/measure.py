import numpy as np


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    """计算mIoU"""
    def get_mIoU(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)# 输出混淆矩阵对角线元素，即每个类别的TP（一维数组形式），加1e-10为陆海亮师哥的计算公式
        union = np.sum(self.confusionMatrix, axis=1) \
            + np.sum(self.confusionMatrix, axis=0) \
            - np.diag(self.confusionMatrix)# 1行FN（axis=1则向内找第二层，即为混淆矩阵每行中的元素值）
                                                    # 2行FP（axis=0则向内找第一层，即为混淆矩阵的每行一维数组）
                                                    # 3行减去重复加的TP部分
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU
    """计算各类别IoU"""
    def get_IoU_list(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)# 输出混淆矩阵对角线元素，即每个类别的TP（一维数组形式）
        union = np.sum(self.confusionMatrix, axis=1) \
            + np.sum(self.confusionMatrix, axis=0) \
            - np.diag(self.confusionMatrix)# 1行FN（axis=1则向内找第二层，即为混淆矩阵每行中的元素值）
                                                    # 2行FP（axis=0则向内找第一层，即为混淆矩阵的每行一维数组）
                                                    # 3行减去重复加的TP部分
        IoU = intersection / union
        return IoU

    """计算平均准确率"""
    def get_accuracy(self):
        # accuracy = TP / ALL
        TP = np.diag(self.confusionMatrix).sum()
        ALL = self.confusionMatrix.sum()
        accuracy = TP / ALL
        return accuracy

    """计算各类别准确率"""
    def get_accuracy_list(self):
        TP = np.diag(self.confusionMatrix)# 各种类对应所有预测正确的像素点数的列表
        ALL = self.confusionMatrix.sum(axis=1)# 各种类对应所有标签值为真的像素点数的列表
        accuracy_list = TP / ALL
        return accuracy_list

    """
        生成混淆矩阵
        此方法不应被外部调用
    """
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    """添加新数据到混淆矩阵中"""
    def addBatch(self, imgPredict, imgLabel):
        imgPredict = imgPredict.cpu()
        imgLabel = imgLabel.cpu()
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    """重置混淆矩阵"""
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

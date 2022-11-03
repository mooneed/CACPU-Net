#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

"""UNet网络模型，pretrained参数暂时无用，为了与其他模型同步"""
class UNet_PReLU(nn.Module):
    def __init__(self, num_classes, num_channels, bilinear=True, pretrained=False):
        super(UNet_PReLU, self).__init__()
        self.name='UNet'
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)# 初始64通道特征图生成, shape: [-1, 64, 256, 256]
        x2 = self.down1(x1)# 最大池化的下采样, shape: [-1, 128, 128, 128]
        x3 = self.down2(x2)# 最大池化的下采样, shape: [-1, 256, 64, 64]
        x4 = self.down3(x3)# 最大池化的下采样, shape: [-1, 512, 32, 32]
        x5 = self.down4(x4)# 最大池化的下采样, shape: [-1, 512, 16, 16]
        x = self.up1(x5, x4)# 双线性插值的上采样, shape: [-1, 256, 32, 32]
        x = self.up2(x, x3)# 双线性插值的上采样, shape: [-1, 128, 64, 64]
        x = self.up3(x, x2)# 双线性插值的上采样, shape: [-1, 64, 128, 128]
        x = self.up4(x, x1)# 双线性插值的上采样, shape: [-1, 64, 256, 256]
        logits = self.outc(x)# 特征图转为预测结果：通过64通道的特征像素以指定1x1卷积核权重转化为4通道的各类别预测分值
        #print("return shape: {}".format(logits.shape))
        return logits# , shape: [-1, 4, 256, 256]

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    """默认0填充且图像尺寸不变的2层卷积核结构"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def get_model(config, **kwargs):
    model = UNet_PReLU(
        num_classes=config.DATASET.NUM_CLASSES,
        num_channels=config.DATASET.NUM_CHANNELS
    )
    return model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
import cv2

from functools import partial
from collections import OrderedDict
import numpy as np

class ConvBnRelu(nn.Module):
    def __init__(self, inc, outc, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 bn_eps=1e-5, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=ksize, stride=stride,
                              padding=pad, dilation=dilation, groups=groups,
                              bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(outc, eps=bn_eps)

        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

        self._weight_init()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)

        if self.has_relu:
            x = self.relu(x)

        return x

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


# for deeplabv3
class _ASPPModule(nn.Module):
    def __init__(self, inc, outc, kernel_size, padding,
                 dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inc, outc,
                                     kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation)
        self.bn = BatchNorm(outc)
        self.relu = nn.ReLU()

        self._weight_init()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _weight_init(self, ):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


class ASPP(nn.Module):
    def __init__(self, inc, stride):
        super(ASPP, self).__init__()

        # inc from dfc_network
        # stride from dfc_network

        # TODO dilation need to be confirmed
        # TODO here modification to adapt the size of input
        # version1 [1,6,12,18]
        if stride == 16:
            self.dilation = [1, 6, 12, 18]
        elif stride == 8:
            self.dilation = [1, 6, 12, 18]

        self.inner_channel = inc // len(self.dilation)

        self.aspp1 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size=1, padding=0, dilation=self.dilation[0])
        self.aspp2 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size=3, padding=self.dilation[1], dilation=self.dilation[1])
        self.aspp3 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size=3, padding=self.dilation[2], dilation=self.dilation[2])
        self.aspp4 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size=3, padding=self.dilation[3], dilation=self.dilation[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, self.inner_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(640, self.inner_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.inner_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)

        self._weight_init()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return self.dropout(x)

    def _weight_init(self):

        for name, child in self.named_children():
            if name not in ['aspp1', 'aspp2', 'aspp3', 'aspp4']:
                for submodule in child.modules():
                    if isinstance(submodule, nn.Conv2d):
                        init.kaiming_normal_(submodule.weight)
                    if isinstance(submodule, nn.BatchNorm2d):
                        init.constant_(submodule.weight, 1)
                        init.constant_(submodule.bias, 0)


# classifier, score_layer
class _FCNHead(nn.Module):
    def __init__(self, inc, outc, inplace=True, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        interc = inc // 4
        self.cbr = ConvBnRelu(inc, interc, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        # self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(interc, outc, kernel_size=1, stride=1, padding=0)
        self._weight_init()

    def forward(self, x):
        x = self.cbr(x)
        # x = self.dropout(x)
        x = self.conv1x1(x)
        return x

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


def _nostride2dilation(m, dilation):
    if isinstance(m, nn.Conv2d):
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation // 2, dilation // 2)
                m.padding = (dilation // 2, dilation // 2)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)

class DeepLabv3_plus(nn.Module):
    # def __init__(self, configs, outc=2, stride=8, pretrained=True):
    def __init__(self, num_classes, num_channels, stride=8, pretrained=True):
        super(DeepLabv3_plus, self).__init__()
        self.input_size = 256
        self.name='deeplabv3_plus-resnet34'
        # self.configs = configs
        backbone = models.resnet34(pretrained=pretrained)
        # outc = configs.nb_classes
        outc = num_classes
        #卷积层 BN（批量归一化）层 激活层 池化层
        self.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn0 = backbone.bn1
        self.relu0 = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False

        self.stride = stride
        # self.spatial_size = [self.configs.input_size // self.stride,
        #                      self.configs.input_size // self.stride]
        self.spatial_size = [self.input_size // self.stride,
                             self.input_size // self.stride]
        if self.stride == 8:
            self.layer3.apply(partial(_nostride2dilation, dilation=2))
            self.layer4.apply(partial(_nostride2dilation, dilation=4))
        elif self.stride == 16:
            self.layer4.apply(partial(_nostride2dilation, dilation=2))

        self.low_level_adaptor = ConvBnRelu(64, 64, 1, 1, 0)
        # concat with output from aspp [128, 64]
        self.aspp_layer = ASPP(inc=512, stride=16)
        self.score_layer = _FCNHead(192, outc=outc)

    def forward(self, x):

        x = self.conv0(x)        #2(1,3,600,600),(1,64,600,600)
        x = self.bn0(x)          #2(1,64,600,600),(1,64,300,300)
        x = self.relu0(x)        #2(1,64,300,300),(1,64,300,300)
        x = self.maxpool(x)      #4(1,64,300,300),(1,64,150,150)

        low_level_fm = self.layer1(x)    #4(1,64,150,150),(1,64,150,150)
        x = self.layer2(low_level_fm)    #8(1,64,150,150),(1,128,75,75)
        x = self.layer3(x)               #16(1,128,75,75),(1,256,75,75)
        x = self.layer4(x)               #16(1,256,75,75),(1,512,75,75)

        aspp_fm = self.aspp_layer(x) #(1,512,75,75),(1,128,75,75)
        #aspp_fm = F.interpolate(aspp_fm, scale_factor=4, mode='bilinear', align_corners=True)
        aspp_fm = F.interpolate(aspp_fm, scale_factor=2, mode='bilinear', align_corners=True)#(1,128,75,75),(1,128,150,150)
        low_level_fm = self.low_level_adaptor(low_level_fm)
        # u = torch.cat([low_level_fm, aspp_fm], dim=1)
        # print(low_level_fm.shape,aspp_fm.shape)
        out = self.score_layer(torch.cat([low_level_fm, aspp_fm], dim=1))#(1,192,150,150),(1,3,150,150)

        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)#(1,3,150,150),(1,3,600,600)

def get_model(config, **kwargs):
    model = DeepLabv3_plus(
        num_classes=config.DATASET.NUM_CLASSES,
        num_channels=config.DATASET.NUM_CHANNELS,
        pretrained=False
    )
    return model

if __name__ =="__main__":
    imgdir = r"E:\documents\1\1.JPG"
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    img = cv2.imread(imgdir, cv2.IMREAD_COLOR)
    print("img", img.shape)
    img = cv2.resize(img, (600, 600))

    img = torch.from_numpy(img).to(device)
    img = torch.transpose(img,0,2)
    input = img.unsqueeze(0).to(torch.float32)

    print("img", img.shape)
    configs= 6
    inputsize = 600
    net = DeepLabv3_plus(configs, inputsize, stride=8, pretrained=True).to(device)
    # print(net)
    net.eval()
    out = net(input)
    print(out)
    print(out.shape)










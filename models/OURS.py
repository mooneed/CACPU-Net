#coding:utf-8
"""
@Time: 2022/4/17 13:15
@Auth: 边缘
@Email: 785314906@qq.com
@Project: PythonProjects
@IDE: PyCharm
@Motto: I love my country as much as myself.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    ECA    an improve excitation channel attention
"""
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

"""UNet网络模型，pretrained参数暂时无用，为了与其他模型同步"""
class UNet(nn.Module):
    def __init__(self, block, num_classes, num_channels, bilinear=True, pretrained=False):
        super(UNet, self).__init__()
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

        self.a1 = block(64,64)
        self.a2 = block(128,128)
        self.a3 = block(256,256)
        self.a4 = block(512,512)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)# 初始64通道特征图生成, shape: [-1, 64, 256, 256]
        x1 = self.a1(x1) + x1
        x2 = self.down1(x1)# 最大池化的下采样, shape: [-1, 128, 128, 128]
        x2 = self.a2(x2) + x2
        x3 = self.down2(x2)# 最大池化的下采样, shape: [-1, 256, 64, 64]
        x3 = self.a3(x3) + x3
        x4 = self.down3(x3)# 最大池化的下采样, shape: [-1, 512, 32, 32]
        x4 = self.a4(x4) + x4
        x5 = self.down4(x4)# 最大池化的下采样, shape: [-1, 512, 16, 16]
        x = self.up1(x5, x4)# 双线性插值的上采样, shape: [-1, 256, 32, 32]
        x = self.up2(x, x3)# 双线性插值的上采样, shape: [-1, 128, 64, 64]
        x = self.up3(x, x2)# 双线性插值的上采样, shape: [-1, 64, 128, 128]
        x = self.up4(x, x1)# 双线性插值的上采样, shape: [-1, 64, 256, 256]
        logits = self.outc(x)# 特征图转为预测结果：通过64通道的特征像素以指定1x1卷积核权重转化为4通道的各类别预测分值
        #print("return shape: {}".format(logits.shape))
        return {
            "feature":x2, # 中间特征, shape: [-1, 128, 128, 128]
            "pred":logits # 最终结果, shape: [-1, 4, 256, 256]
        }


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


"""PointRend部分"""
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    B, _, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)

    if not training:
        H_step, W_step = 1 / H, 1 / W
        N = min(H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)

        points = torch.zeros(B, N, 2, dtype=torch.float, device=device)
        points[:, :, 0] = W_step / 2.0 + (idx  % W).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // W).to(torch.float) * H_step
        return idx, points
        
    over_generation = torch.rand(B, k * N, 2, device=device)
    over_generation_map = point_sample(mask, over_generation, align_corners=False)

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]

    importance = over_generation.view(-1, 2)[idx.view(-1), :].view(B, int(beta * N), 2)
    coverage = torch.rand(B, N - int(beta * N), 2, device=device)
    return torch.cat([importance, coverage], 1).to(device)

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

class PointHead(nn.Module):
    def __init__(self, in_c=533, num_classes=4, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, bb_feature, bb_pred):
        if not self.training:
            return self.inference(x, bb_feature, bb_pred)
        # 在精细预测图上选最不确定的点, shape [-1, N, 2], N个点的xy坐标
        points = sampling_points(bb_pred, (x.shape[-1] // 2) ** 2, self.k, self.beta)
        # 提取粗糙预测结果上对应点预测结果, shape [-1, 4, N]
        coarse = point_sample(bb_pred, points, align_corners=False)
        # 提取精细特征图上对应点特征值, shape [-1, 4, N]
        fine = point_sample(bb_feature, points, align_corners=False)
        # 特征合并, shape [-1, 132, N]
        feature_representation = torch.cat([coarse, fine], dim=1)
        # 输入到卷积核1的卷积层中，预测结果
        point_pred = self.mlp(feature_representation)

        return {"point_pred": point_pred, "points": points}

    @torch.no_grad()
    def inference(self, x, bb_feature, bb_pred):
        num_points = 8096

        while bb_pred.shape[-1] != x.shape[-1]:
            bb_pred = F.interpolate(bb_pred, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(bb_pred, num_points, training=self.training)

            coarse = point_sample(bb_pred, points, align_corners=False)
            fine = point_sample(bb_feature, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = bb_pred.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            bb_pred = (bb_pred.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        return {"fine": bb_pred}

"""PointRend"""
class PointRend(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        result = self.backbone(x)
        result.update(# 更新预测结果
            self.head(
                x, # 原始数据输入
                result["feature"], # 粗糙特征
                result["pred"] # 预测结果
            )
        )
        return result

def get_model(config, **kwargs):
    backbone = UNet(
        block=ECABasicBlock,
        num_classes=config.DATASET.NUM_CLASSES,
        num_channels=config.DATASET.NUM_CHANNELS
    )
    head = PointHead(in_c=132)
    model = PointRend(
        backbone=backbone,
        head=head,
    )
    return model
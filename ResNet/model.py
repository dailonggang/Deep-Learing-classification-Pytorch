import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    """
    对应18层和34层的网络结构
    """
    # 定义residual结构的卷积核个数是否发生变化，可以看到18层和34层对应卷积核分别为[64, 128, 256, 512]，并未发生变化，所以设为1
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        :param in_channel: 输入通道
        :param out_channel: 输出通道
        :param stride: 步长
        :param downsample: 对应虚线残差结构，因为每一层对应的卷积核发生的变化，需要使用虚线残差结构进行维度变化
        :param kwargs: 允许你将不定长度的键值对, 作为参数传递给一个函数。
        """
        super(BasicBlock, self).__init__()  # 继承父类
        # 下面就是对应18层和34层的residual结构，对应两个3×3卷积，使用归一化和relu激活函数
        # 这里stride=1时，对应实线残差结构，stride=2时对应虚线残差结构，因为输入矩阵的高和宽减半，使用BN层就没有必要使用bias
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 定义residual残差边是否采用1×1的卷积
        self.downsample = downsample

    def forward(self, x):
        # 输入x传递给捷径分支
        identity = x
        # 如果downsample不为None, 那么进行捷径分支的卷积操作
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 先进行主分支和残差结构的和操作，再进行激活操作
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    对应50层、101层和152层的网络结构
    """
    # 这里以50层为例，可以看到residual结构的第三层卷积核的个数是前两层的四倍，所以这里expansion设为4
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # 实线和虚线步长不定
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # 输出通道变为原来的四倍
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        """
        :param block: 对应为BasicBlock还是Bottleneck
        :param blocks_num: 代表conv2x-conv5x所使用残差结构的数目，是一个列表数据结构。例如在34-layer的网络结构中对用为[3, 4, 6, 3]
        :param num_classes: 对应类别数
        :param include_top: 方便以后搭建其他网络
        :param groups: 分组卷积
        :param width_per_group:
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # max pool后的通道数

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 通过_make_layer()函数生成conv2_x~conv5_x
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 对卷积层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        :param block: 对应为BasicBlock还是Bottleneck
        :param channel: 对应残差结构第一层对应的通道数，例如开始都为64
        :param block_num: 代表conv2x-conv5x所使用残差结构的数目
        :param stride: 步长
        :return:
        """
        downsample = None

        # 这里对于18和34层的网络而言，两个条件都不满足所以直接跳过if语句
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # 定义一个空列表
        layers = []
        # 将对应block的residual结构添加进layers
        layers.append(block(self.in_channel, channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        # 按照对应block结构进行改变
        self.in_channel = channel * block.expansion

        # 利用循环实现每层对应的个数
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # 利用非关键字参数传入 nn.Sequential,得到相应的layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

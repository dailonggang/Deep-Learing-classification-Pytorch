import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),   # 这里的填充应该是左上各补一行零，右下补两行零，但是最后会有小数出现，会自动舍弃一行零。input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),                           # output[2048]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),                                  # output[2048]
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),                           # output[1000]
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # pytorch的tensor通道排列顺序为[batch, channel, height, weight].
        # start_dim=1也就是从第一维度channel开始展平成一个一维向量
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 权重初始化，现阶段pytorch版本自动进行权重初始化
    def _initialize_weights(self):
        # 遍历定义的各个模块
        for m in self.modules():
            # 如果模块为nn.Conv2d，就用kaiming_normal_()方法对权重进行初始化，如果偏置为空，就用0进行初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果模块为全连接层，就用normal_()正态分布来给权重赋值，均值为0，方差为0.01
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


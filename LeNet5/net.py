import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):  # 继承nn.Module
    # 初始化网络
    def __init__(self):
        # 利用super().__init__()把类LeNet的对象self转换为类nn.Module的对象，然后“被转换”的类的nn.Module对象调用自己的init函数
        # 简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西
        super(LeNet, self).__init__()  # 对继承自父类的属性进行初始化，也就是说子类继承了父类的所有属性和方法
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)


    def forward(self, x):
        """
        N = (W-F+2P)/S+1
        N:输出大小
        W:输入大小
        F:卷积核大小
        P:填充物大小
        S:步长大小
        """
        x = F.relu(self.conv1(x))    # input(1, 32, 32) output(6, 28, 28)    [channel, w, h]
        x = self.pool1(x)            # output(6, 14, 14)
        x = F.relu(self.conv2(x))    # output(16, 10, 10)
        x = self.pool2(x)            # output(16, 5, 5)
        x = F.relu(self.conv3(x))    # output(120, 5, 5)
        x = self.flatten(x)          # output(120)
        x = F.relu(self.fc1(x))      # output(84)
        x = self.output(x)              # output(10)
        return x


if __name__ == '__main__':
    x = torch.rand([1, 1, 32, 32])
    model = LeNet()
    y = model(x)

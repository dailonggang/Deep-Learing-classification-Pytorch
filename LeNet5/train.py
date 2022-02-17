import torch
from torch import nn
from net import LeNet
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据转化为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用model里面定义的模型，将模型数据转到GPU
net = LeNet().to(device)

# 定义一个损失函数（交叉熵损失函数）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器（SGD：随机梯度下降）
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

# 学习率，每隔10轮变为原来的0.1倍
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 定义训练函数
def train(dataloader, net, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向传播
        X, y = X.to(device), y.to(device)
        output = net(X)
        cur_loss = loss_fn(output, y)
        # torch.max()函数返回两个值，第一个是具体的value(用下划线表示)，第二个是value所作在的index(用pred表示)
        # dim=1表示输出所在行的最大值
        _, pred = torch.max(output, dim=1)

        # output.shape[0]为一个批次
        cur_acc = torch.sum(y==pred)/output.shape[0]

        # 反向传播
        # 梯度置零，也就是把loss关于weight的导数变为0
        optimizer.zero_grad()
        # 反向传播求解梯度
        cur_loss.backward()
        # 更新权重参数
        optimizer.step()

        # item()返回loss的值，叠加后算出总loss
        loss += cur_loss.item()
        # item()返回acc的值，叠加后算出总acc
        current += cur_acc.item()
        n = n + 1
    # 除以mini-batch的数量，取平均值
    print("train_loss" + str(loss/n))
    print("train_acc" + str(current/n))


# 定义验证函数
def val(dataloader, net, loss_fn):
    # 转为验证模式
    net.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，模型参数无需更新
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = net(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)

            cur_acc = torch.sum(y==pred)/output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss" + str(loss/n))
        print("val_acc" + str(current/n))
        return current/n


# 开始训练
epoch = 10
max_acc = 0
for i in range(epoch):
    print(f'epoch{i+1}\n---------------')
    train(train_dataloader, net, loss_fn, optimizer)
    a = val(test_dataloader, net, loss_fn)
    # 保存最好的模型权重
    if a > max_acc:
        folder = 'save_model'
        # 如果文件夹不存在则进行创建
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = a
        print('save best model')
        # 进行模型参数保存
        torch.save(net.state_dict(), 'save_model/best_model.pth')
print('Over!!!')

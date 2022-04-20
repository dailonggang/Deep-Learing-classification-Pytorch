import torch
from net import LeNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# 数据转化为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据集加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用model里面定义的模型，将模型数据转到GPU
net = LeNet().to(device)

# 将train.py里预训练的参数权重加载到模型
net.load_state_dict(torch.load("D:/pycode/CNN/Lenet/save_model/best_model.pth"))

# 获取测试结果
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9 ', ]


# 把tensor转化Image，方便可视化
show = ToPILImage()

# 进入验证,这里只进行5张照片的验证，有需要可将5换为test_dataset进行测试集的推理
for i in range(5):
    # 对应图片和标签
    X, y = test_dataset[i][0], test_dataset[i][1]
    # 将图片进行展示
    show(X).show()

    # 扩展张量维度为4维.torch.unsqueeze()表示加维度，torch.squeeze()表示减维度
    # variable()可以将tensor转为Variable的形式，因为tensor不能反向传播，variable可以反向传播。requires_grad决定是否求导
    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
    # 反向传播时不进行求梯度，节省内存空间
    with torch.no_grad():
        pred = net(X)
        # print(pred[0])
        # 得到预测类别中最高的哪一类，以及真实值.torch.argmax()返回维度上张量最大值的索引
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted: "{predicted}", actual:"{actual}"')

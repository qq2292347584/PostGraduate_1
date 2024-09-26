import torch.nn as nn
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from torch import optim
from model import LeNet

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
) # Normalize 是一个标准化的过程

trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                          shuffle=True, num_workers=0)
# shuffle 表示是否打乱顺序  num_workers 表示载入数据的线程数 一般在windows系统下设置为0

# 10000张测试图片
testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                       download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=0)
test_data_iter = iter(testloader) # 将其转化为迭代器
test_image, test_label = next(test_data_iter)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss() # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr = 0.001) # 定义优化器

for epoch in range(5): # 训练的轮次

    running_loss = 0.0
    for step, data in enumerate(trainloader, start = 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # 将历史损失梯度清零
        # 为什么每计算一个batch, 就需要调用一次optimizer.zer_grad()函数
        # 如果不清除历史梯度，就会对计算的历史梯度进行累加
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward() # 反向传播
        optimizer.step() # 通过优化器对参数进行更新

        running_loss +=loss.item()
        if step % 500 == 499: # 每隔500步打印一次信息
            with torch.no_grad():
                outputs = net(test_image) # [batch, 10]
                predict_y = torch.max(outputs, dim = 1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss:%.3f test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')
save_path = './LeNet.pth'
torch.save(net.state_dict(), save_path)
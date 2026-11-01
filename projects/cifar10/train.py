import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from datetime import datetime

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device:{device}')

epochs=50

print('-------------------------------数据准备----------------------------------------')

better_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),#随机裁剪，先将图片放大到40*40，然后随机裁剪出32*32的图片(优化模型的泛化能力)
    transforms.RandomHorizontalFlip(),#随机50%概率水平翻转（优化模型的泛化能力）
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
#ToTensor()：把图片转换成 PyTorch 张量,像素值归一化到 0～1,Normalize：继续调整张量的数值
# Normalize它对每个像素执行：output = (input - mean)/std,也就是把数值从[0, 1] → [-1, 1]
# 因为归一化是按通道进行的。MNIST 只有一个灰度通道，所以列表里只有一个数。
# 因为是三通道 RGB 图片，通常需要分别给三个通道设置参数，例如：Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

path='D:/ai/cifar10_data/'#数据集下载后保存的目录

train_data=torchvision.datasets.CIFAR10(path,train=True,transform=better_transform,download=True)
test_data=torchvision.datasets.CIFAR10(path,train=False,transform=transform,download=True)
#torchvision.datasets中的torchvision.datasets库中提供了CIFAR10数据集的下载地址

split_generator=torch.Generator().manual_seed(42)
#random_split 主要分的是“索引”，不是把图片真正复制成两份。
#val_set 是通过 random_split 得到的 Subset 对象，它本身主要负责保存索引；真正执行 transform 的是它里面的 dataset。

train_set,val_set=random_split(train_data,[45000,5000],generator=split_generator)
val_set.dataset=torchvision.datasets.CIFAR10(path,train=True,transform=transform,download=False)
test_set=test_data

image,label=train_data[0]
print("训练集数量：",len(train_set))
print("验证集数量：",len(val_set))
print("测试集数量：",len(test_set))
print("图片形状：",image.shape)
print("标签：",label)
print('图片标签:',label)


train_loader=torch.utils.data.DataLoader(train_set,batch_size=256,shuffle=True)
val_loader=torch.utils.data.DataLoader(val_set,batch_size=64,shuffle=False)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=False)
#shuffle：是否打乱数据集

print('-------------------------------数据准备完成,开始构建模型------------------------------------')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.pool=torch.nn.MaxPool2d(
            kernel_size=2,stride=2)
        self.flatten=torch.nn.Flatten()
        self.fc1=torch.nn.Linear(512 * 4 * 4, 512)
        self.fc2=torch.nn.Linear(512, 64)
        self.fc3=torch.nn.Linear(64, 10)

        self.bn1=torch.nn.BatchNorm2d(16)
        #BatchNorm2d(32) 会分别处理这 32 个通道，让每个通道的数值分布更稳定，并且每个通道都有可以学习的缩放和偏移参数。
        self.bn2=torch.nn.BatchNorm2d(32)
        self.bn3=torch.nn.BatchNorm2d(64)
        self.bn4=torch.nn.BatchNorm2d(128)
        self.bn5=torch.nn.BatchNorm2d(256)
        self.bn6=torch.nn.BatchNorm2d(512)

        self.relu=torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(p=0.3)
        #训练时30%概率关闭激活值
        #它防止多个神经元过度依赖彼此，迫使网络学习更稳健、更分散的特征，从而缓解过拟合。


#经典结构：
#Conv → BN → ReLU → Conv → BN → ReLU → Pool
#Conv → BN → ReLU → Conv → BN → ReLU → Pool
#Conv → BN → ReLU → Conv → BN → ReLU → Pool


    def forward(self,x):
        x=self.conv1(x)#16*32*32
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)#32*32*32
        x=self.bn2(x)
        x=self.relu(x)
        x=self.pool(x)#32*16*16

        x=self.conv3(x)#64*16*16
        x=self.bn3(x)
        x=self.relu(x)
        x=self.conv4(x)#128*16*16
        x=self.bn4(x)
        x=self.relu(x)
        x=self.pool(x)#128*8*8

        x=self.conv5(x)#256*8*8
        x=self.bn5(x)
        x=self.relu(x)
        x=self.conv6(x)#512*8*8
        x=self.bn6(x)
        x=self.relu(x)
        x=self.pool(x)#512*4*4


        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc3(x)
        return x

net=Net().to(device)
print('模型的结构：',net)


optimizer=torch.optim.AdamW(net.parameters(),lr=0.001,weight_decay=0.0001)
#AdamW优化器是Adam优化器的一个变体，它在更新权重时引入了权重衰减（weight decay）机制，有助于防止过拟合（模型对某些错误的东西越来越自信）。
#相比于传统的L2正则化，AdamW在计算梯度时不会将权重衰减直接应用于梯度，而是在更新权重时单独应用权重衰减，从而提高了训练的稳定性和模型的泛化能力。
criterion=torch.nn.CrossEntropyLoss()
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
#CosineAnnealingLR 是一种学习率调度器，它会根据余弦函数的变化规律来调整学习率。在训练过程中，学习率会逐渐减小,形成一个类似于余弦曲线的前半段变化趋势。
#这里T-max表示一个余弦周期的1/2，也就是从50循环中，学习率逐渐减小到0.
#余弦退火


print('-------------------------------模型构建完毕，开始训练模型---------------------------------')

def evaluate(model, data_loader, criterion,device):
    model.eval()#dropout,batchnorm，在评估模式下，dropout层不会随机丢弃神经元，batchnorm层会使用训练时的均值和方差，而不是当前批次的均值和方差。
    total_loss = 0.0
    correct = 0#正确样本数
    total = 0#总样本数

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            #forward
            loss = criterion(outputs, labels)#这一批次的平均损失
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            # outputs.argmax(dim=1) 返回每个样本预测的类别索引，dim=1表示按行取最大值的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total

best_val_acc=0.0
best_model_path=f'cifar10_best_model_{datetime.now():%m%d_%H%M}.pth'

for epoch in range(epochs):
    net.train()
    train_loss=0.0
    train_correct=0

    process_bar=tqdm(train_loader,desc=f'Epoch {epoch+1}/{epochs}',unit='batch')
    for i,(images, labels) in enumerate(process_bar):#batch循环

        images=images.to(device)
        labels=labels.to(device)

        #训练的总过程:前向传播→计算损失→反向传播→更新参数

        optimizer.zero_grad()#清空上一轮梯度
        outputs=net(images)
        loss=criterion(outputs,labels)#crizerion
        loss.backward()
        optimizer.step()#optimizer

        train_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()

    #训练集的平均损失和准确率
    train_loss_acc,traincorrect_acc=train_loss/len(train_loader.dataset),train_correct/len(train_loader.dataset)
    #验证集的平均损失和准确率
    val_loss_acc,val_correct_acc=evaluate(net,val_loader,criterion,device)

    scheduler.step()#更新学习率(优化)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_acc:.4f}, Train Acc: {traincorrect_acc:.4f}')
    print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss_acc:.4f}, Val Acc: {val_correct_acc:.4f}')

    if val_correct_acc > best_val_acc:
        best_val_acc = val_correct_acc
        torch.save(net.state_dict(), best_model_path)
        #推荐的方式是保存模型的 state_dict，这样可以更灵活地加载到不同架构中
        print('目前最好的模型已保存，准确率为：{:.4f}'.format(best_val_acc))

print('-------------------------------训练循环结束，接下来是测试模型时间--------------------------------')

net.load_state_dict(torch.load(best_model_path, map_location=device))

test_loss_acc,test_correct_acc=evaluate(net,test_loader,criterion,device)
print(f'该模型的测试结果是:Test Loss: {test_loss_acc:.4f}, Test Acc: {test_correct_acc:.4f}')


print('----------------------------------------测试结束------------------------------------------')

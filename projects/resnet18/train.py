import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from datetime import datetime
from model import ResNet18


def main():

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device:{device}')

        #见ResNet定义



    ####-------复用cnn模型---------



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


    train_loader=torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_set,batch_size=64,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)
    #shuffle：是否打乱数据集


    net=ResNet18().to(device)
    print('模型的结构：',net)

    optimizer=torch.optim.SGD(net.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    #net.parameters() 必须在建 optimizer 的那一刻就传进去
    #momentum：v ← 0.9 × v + grad（这里的 grad 已包含 SGD 的 L2 惩罚项）。
    #weight_decay=5e-4 —— 权重衰减（L2 正则）,grad ← grad + 5e-4 × w
    #SGD 的 L2 项进入梯度后还会进入动量，不能把整个带动量的更新简单等同于参数乘固定因子。
    criterion=torch.nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)


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

    best_val_acc=-1.0
    best_model_path=f'cifar10_resnet_best_model_{datetime.now():%m%d_%H%M}.pth'

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


if __name__ == "__main__":
    main()

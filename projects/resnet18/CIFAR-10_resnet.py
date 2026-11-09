import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from datetime import datetime

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device:{device}')

class basicblock(torch.nn.Module):
    
    #ResNet中对于50层以下的构建块采用的是BasicBlock，而大于50的深层则采用的是Bottleneck

    expansion=1
    #expansion 是给 Bottleneck 用的“通道放大倍数”。
    #BasicBlock 里输入输出通道数一致，所以是 1（ResNet-18/34 用 BasicBlock，ResNet-50+ 用 Bottleneck，那里的 expansion=4）。
    
    def __init__(self,inchannels,outchannels,stride=1):
    #参数:	
    #in_channels	输入特征图的通道数
    #out_channels	主分支卷积使用的输出通道数
    #stride=1	默认不缩小高、宽；传入 2 时降采样
        super(basicblock,self).__init__()


        self.conv1=torch.nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=stride,padding=1,bias=False)
        #下采样只发生在 conv1 的 stride 上,conv2 永远是 stride=1。这是 ResNet 的标准设计。
        #卷积偏置(bias)相当于在结果上再加一个可学习的数。
        #bias=False 表示不使用偏置项，因为后面会接 BatchNorm2d，BN 自己有可学习的缩放和平移，因此通常可以省去卷积偏置。它和残差相加是两件不同的事。
        self.bn1=torch.nn.BatchNorm2d(outchannels)
        self.relu=torch.nn.ReLU(inplace=True)
        #ReLU接受一个名为inplace的参数，该参数的默认值为False。
        #当inplace设置为True时，激活函数会直接在输入数据上进行修改，而不是创建一个新的对象来存储输出结果。这样做的好处是可以节省内存的使用，因为不需要为输出结果分配新的空间。此外，原地操作（in-place operation）还可以减少内存的申请和释放时间，提高效率。
        self.conv2=torch.nn.Conv2d(outchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=torch.nn.BatchNorm2d(outchannels)


        self.shortcut=torch.nn.Sequential()
        #nn.Sequential()是一个有序的容器，里面的模块会按照传入的顺序依次执行。
        if stride!=1 or inchannels!=outchannels:
            self.shortcut=torch.nn.Sequential(
                torch.nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=stride,bias=False),
                torch.nn.BatchNorm2d(outchannels)
            )
            #残差相加要求两边形状一致，
            #一维卷积核在这里的作用不是提特征，而是对齐形状，让 out + identity 能做逐元素相加.
        

    def forward(self,x):
        identity=self.shortcut(x)
        #放在前面，先把残差连接的x做完，后面再做主分支的卷积，后面x会改变。

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)

        x+=identity
        x=self.relu(x)
        #标准残差是整体加起来再relu激活

        return x

class ResNet(torch.nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        #这里num_blocks后面会传入[2,2,2,2]

        #参数:
        #block: 构建块类型，通常是 BasicBlock 或 Bottleneck。
        #num_blocks: 每个阶段的构建块数量，通常是一个列表，例如 [2, 2, 2, 2] 表示每个阶段有 2 个构建块。
        #num_classes: 分类任务的类别数，默认是 10。

        super(ResNet,self).__init__()

        self.inchannels=64
        self.conv1=torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=torch.nn.BatchNorm2d(64)
        self.relu=torch.nn.ReLU(inplace=True)#N,64,32,32

        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)#N,64,32,32
        #名字前面的单下划线 _ 是一种约定，表示“主要供内部使用的方法”。
        #_make_layer 是一个辅助函数，用于创建一个由多个构建块组成的层。它会根据传入的参数生成指定数量的构建块，并将它们按顺序连接起来，形成一个完整的网络层。
        
        ###这里的每个层都要调用开始定义的残差结构，如这个stride是1，那么之后就是[1,1,1,...],若像下面一样stride是2，则[2,1,1,..]
        
        #参数：
        #block: 构建块类型，通常是 BasicBlock 或 Bottleneck。
        #out_channels: 当前层的输出通道数。
        #num_blocks: 当前层的构建块数量。
        #stride: 当前层的第一个构建块的步幅，通常用于下采样。(若2)
        
        #下采样：下采样是指将图像或特征图的空间分辨率按比例降低（如从 H×W 变为 H/s×W/s），常用于减少计算量、扩大感受野并提取更抽象的特征，典型方法包括最大池化、平均池化和步长卷积等。
        
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)#分辨率减半,B,128,16,16
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)#分辨率减半,B,256,8,8
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)#分辨率减半,B,512,4,4

        self.avgpool=torch.nn.AdaptiveAvgPool2d((1,1))#B,512,1,1
        #AdaptiveAvgPool2d 是一种自适应平均池化层，它可以将输入特征图的空间尺寸调整为指定的输出尺寸 (H_out, W_out)，而不需要手动计算池化窗口的大小和步幅。无论输入特征图的大小如何变化，AdaptiveAvgPool2d 都能确保输出特征图具有固定的空间尺寸，从而简化了网络设计。
        self.fc=torch.nn.Linear(512*block.expansion,num_classes)#512->10


    def _make_layer(self,block,outchannels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
            #第一个残差块(num_blocks)的步幅为 stride，后续的残差块步幅为 1。
        layers=[]
        for stride in strides:#[stride,1,1,...]
            layers.append(block(self.inchannels,outchannels,stride))

            self.inchannels=outchannels*block.expansion
            #更新输入通道数，以便下一层使用。
            #这一行是为了兼容 Bottleneck 写的。现在用 BasicBlock，* 1 等于没乘，可以直接忽略；等以后换 ResNet-50 才真正用得上。
            #同理 self.fc = nn.Linear(512 * block.expansion, ...) 也是这个原因。

        return torch.nn.Sequential(*layers)
        
    ###我们来简化一下上述过程，可以理解为：
    ##self.layer1 = nn.Sequential(
    #BasicBlock(64, 64, 1),
    #BasicBlock(64, 64, 1),）

    ##self.layer2 = nn.Sequential(
    #BasicBlock(64, 128, 2),
    #BasicBlock(128, 128, 1),)

    ##self.layer3 = nn.Sequential(
    #BasicBlock(128, 256, 2),
    #BasicBlock(256, 256, 1),)

    ##self.layer4 = nn.Sequential(
    #BasicBlock(256, 512, 2),
    #BasicBlock(512, 512, 1),)

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)#B,512,4,4
        x=self.avgpool(x)#B,512,1,1
        x=torch.flatten(x,1)#B,512
        #torch.flatten：把x的高维特征图"拉平"成一个一维向量。1 表示从第 1 维开始拉平（第 0 维是 batch，保留不动）
        #torch.flatten(input, start_dim=0, end_dim=-1)
        #start_dim：从哪一维开始拉平，默认 0
        #end_dim：拉到哪一维结束，默认 -1（最后一维）

        #也可以直接：self.flatten = nn.Flatten()
        #torch.nn.Flatten(start_dim=1, end_dim=-1)

        x=self.fc(x)

        return x


def ResNet18():
    return  ResNet(basicblock,[2,2,2,2])
    #见ResNet定义


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

####-------复用cnn模型---------


def main():

    epochs=50

    print('-------------------------------数据准备----------------------------------------')

    better_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),#随机裁剪，先将图片放大到40*40，然后随机裁剪出32*32的图片(优化模型的泛化能力)
        transforms.RandomHorizontalFlip(),#随机50%概率水平翻转（优化模型的泛化能力）
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

    #这里需要注意，RandomCrop and RandomHorizontalFlip 均是在cpu上进行，因为串行，所以每次gpu训练必须得等cpu处理完数据再进行，会导致训练变慢很多
    #这是为什么后面DataLoader需要开num_worker等参数

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
    #val_set 是通过 random_split 得到的 Subset(子集) 对象，它本身主要负责保存索引；真正执行 transform 的是它里面的 dataset。

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


    train_loader=torch.utils.data.DataLoader(train_set,batch_size=128,shuffle=True,num_workers=4, pin_memory=True, persistent_workers=True)
    
    val_loader=torch.utils.data.DataLoader(val_set,batch_size=64,shuffle=False)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)
    #shuffle：是否打乱数据集


    net=ResNet18().to(device)
    print('模型的结构：',net)

    optimizer=torch.optim.SGD(net.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    #net.parameters() 必须在建 optimizer 的那一刻就传进去
    #momentum动量：v ← 0.9 × v + grad  ,0.9 表示"保留 90% 的旧速度，加入 10% 的新梯度"
    #weight_decay=5e-4 —— 权重衰减（L2 正则）,grad ← grad + 5e-4 × w        
    # PyTorch 的 SGD 是这么实现的（把衰减塞进梯度）w ← w - lr × grad,也就是每步额外乘一个因子 (1 - lr × wd) = 1 - 0.1×5e-4 = 0.99995。
    criterion=torch.nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)


    print('-------------------------------模型构建完毕，开始训练模型---------------------------------')


    best_val_acc=0.0
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

if __name__=='__main__':
    main()
#
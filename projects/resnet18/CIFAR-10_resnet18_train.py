import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import Dataset

from datetime import datetime

from model_resnet18 import ResNet18

import random
import numpy as np
#在pytorch的底层,一直隐藏调用numpy,numpy贯穿深度学习始终


def set_seed(seed):
    random.seed(seed)                # 固定 Python 内置的随机性
    np.random.seed(seed)             # 固定 Numpy 的随机性
    torch.manual_seed(seed)          # 固定 PyTorch CPU 的随机性
    torch.cuda.manual_seed_all(seed) # 固定 PyTorch 所有 GPU 的随机性
#将深度学习中所有可能用到的随机数全部设定种子，保证其随机性固定，训练更稳定，更方便控制变量对比

    torch.backends.cudnn.benchmark = True 
    #其是 PyTorch 中的一个设置，用于启用 cuDNN 的自动优化功能。它会根据硬件和输入数据的大小，选择最优的卷积算法，从而提升模型的运行效率。
    #但如果是严谨的学术论文，设置为False


def worker_init_fn(work_id):
    worker_seed = torch.initial_seed() % 2**32
    #固定模板，记住就行，不必深究
    np.random.seed(worker_seed)
    random.seed(worker_seed)
#在num_workers的多进程中，各子进程实际上并不关心主进程seed的设置，所以要想固定，也要把子进程锁死seed
#给每一个id的人发一个seed


class TransformSubset(Dataset):
    #TransformSubset 这个类继承了 Dataset 这个类。
    def __init__(self, base, subset, transform):
        #base是数据集，subset是子集或者说是一个存放索引的列表（参考后面代码），transform是对base的处理操作
        self.base, self.subset, self.transform = base, subset, transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        img, y = self.base[self.subset[i]]
        #base本身是个列表，它里面包含图片和标签,详细参考后面print数量前的操作
        #self.subset[i]返回的就是一个索引数字
        return self.transform(img), y
    #以双下划线开头和结尾的方法（如 __init__、__len__、__getitem__）被称为魔法方法
    #它们是 Python 语言内置的“钩子”，你不能直接调用它们，而是由 Python 解释器在特定操作下自动调用。
        
    ###这是编程里的重写(override)，重写了该函数父类dataset里的方法
    ##Dataset 父类其实长这样：
    #class Dataset:
    #    def __getitem__(self, index):
    #       raise NotImplementedError  # 意思是：别用我，我什么都没写，去你子类里写！
    #    def __len__(self):
    #        raise NotImplementedError
    ###父类 Dataset 只是规定了一个“契约”：凡是我的儿子（子类），都必须有这两个方法。但父类自己不知道你将来是要做图像分类、语音识别还是文本翻译，它没法提前写死逻辑，所以它就留了个“空壳”。

    ###该改写在后文dataloader会被触发，下文中train_set和val_set调用了该类，即为该类的实例，继承该方法，dataloader会获取train_set[i],触发改动的方法
    ###当然test_set没有调用该类，则为原始dataset方法
    ###该思想在函数中称为多态。



def evaluate(model, data_loader, criterion,device):
    model.eval()#dropout,batchnorm，在评估模式下，dropout层不会随机丢弃神经元，batchnorm层会使用训练时的均值和方差，而不是当前批次的均值和方差。
    total_loss = 0.0
    correct = 0#正确样本数
    total = 0#总样本数

    with torch.inference_mode():
    #with后跟语句可以视为临时应用，常用于文件操作，结束后会自动关闭

    #在验证集和测试集上，我们不需要计算梯度
    #torch.inference_mode()（推理模式）是torch.no_grad()的上位替代
    #它直接把记录计算图相关的底层变量全禁用了，内存消耗更少，运行速度微快。

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


def main():

    set_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device:{device}')

    epochs=100

    print('-------------------------------数据准备----------------------------------------')

    better_transform=transforms.Compose([
        transforms.RandomCrop(32,padding=4),#随机裁剪，先将图片放大到40*40，然后随机裁剪出32*32的图片(优化模型的泛化能力)
        transforms.RandomHorizontalFlip(),#随机50%概率水平翻转（优化模型的泛化能力）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        #cifar10真实统计数据,直接用.
        transforms.RandomErasing(p=0.5)#随机擦除一块矩形区域,强迫模型不要只盯着局部特征看，而是学会看整体,它的输入必须是 Tensor，不能是 PIL 图片
    ])
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    #ToTensor()：把图片转换成 PyTorch 张量,像素值归一化到 0～1,Normalize：继续调整张量的数值
    # Normalize它对每个像素执行：output = (input - mean)/std,也就是把数值从[0, 1] → [-1, 1]
    # 因为归一化是按通道进行的。MNIST 只有一个灰度通道，所以列表里只有一个数。
    # 因为是三通道 RGB 图片，通常需要分别给三个通道设置参数，例如：Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

    path='D:/ai/cifar10_data/'#数据集下载后保存的目录
    
    base = torchvision.datasets.CIFAR10(path, train=True, transform=None, download=True)
    #transform=None,确保base是纯净的pil图片,没有经过transform处理,方便在下面set设置时再进行处理
    
    split_generator=torch.Generator().manual_seed(42)
    train_index,val_index=random_split(range(len(base)),[45000,5000],generator=split_generator)
    #range(len(base))是从0到49999共50000个数字，也就是索引，分后方便传给__getitem__
    train_set=TransformSubset(base,train_index,better_transform)
    val_set=TransformSubset(base,val_index,transform)
    #random_split 主要分的是“索引”，不是把图片真正复制成两份。

    test_set=torchvision.datasets.CIFAR10(path,train=False,transform=transform,download=True)
    #torchvision.datasets中的torchvision.datasets库中提供了CIFAR10数据集的下载地址

    image,label=train_set[0]
    print("训练集数量：",len(train_set))
    print("验证集数量：",len(val_set))
    print("测试集数量：",len(test_set))
    print("图片形状：",image.shape)
    print("随机标签：",label)


    train_loader=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True,
                                            num_workers=4,pin_memory=True,persistent_workers=True,
                                            prefetch_factor=4,worker_init_fn=worker_init_fn)
    #在深度学习训练中，GPU 的计算速度非常快，但如果 CPU 给 GPU 喂数据（读图、裁剪、归一化）的速度跟不上，GPU 就会闲等，导致训练极慢。
    #num_workers 4进程数据加载,pin_memory=True锁页内存,显著加快 CPU 到 GPU 的数据传输速度。只要你用 GPU 训练，强烈建议开启
    #persistent_workers=True这 4 个子进程在 epoch 之间不会被销毁，而是保持存活，等待下一个 epoch 继续工作。
    #prefetch_factor 每个 worker 在后台准备数据的量。（总共提前准备了 4 个 worker * 4 = 16 个 batch 的数据）
    #worker_init_fn见上文函数定义.

    val_loader=torch.utils.data.DataLoader(val_set,batch_size=32,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    #shuffle：是否打乱数据集


    net=ResNet18().to(device)
    print('模型的结构：',net)

    optimizer=torch.optim.SGD(net.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)
    #net.parameters() 必须在建 optimizer 的那一刻就传进去
    #momentum：v ← 0.9 × v + grad（这里的 grad 已包含 SGD 的 L2 惩罚项）。
    #weight_decay=5e-4 —— 权重衰减（L2 正则）,grad ← grad + 5e-4 × w
    #SGD 的 L2 项进入梯度后还会进入动量，不能把整个带动量的更新简单等同于参数乘固定因子。
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    #让模型保持90%的自信，避免过拟合，把多余的注意力分散给其他类别，提升模型在测试集上的泛化能力。
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    scaler=torch.amp.GradScaler('cuda')


    print('------------------------------------------模型构建完毕，开始训练模型------------------------------------------------')

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

            with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                outputs=net(images)
                loss=criterion(outputs,labels)#crizerion
                #scaler的定义在优化器后
                #默认情况下，PyTorch 里的模型参数和计算都是 32 位浮点数（FP32），这叫“单精度”
                #但现在的NVIDIA 显卡里面有专门的“Tensor Core”硬件，做 16 位浮点数（FP16）的计算速度极快，而且占用的显存只要原来的一半。
                #AMP（自动混合精度）的作用:让 GPU 在算矩阵乘法（卷积、全连接）这种大计算量任务时用 FP16（图快、省显存）
                #在累加、更新参数等需要高精度的操作时，自动切回FP32

            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            #参数必加
            scaler.update()# 更新缩放因子


            process_bar.set_postfix(loss=f"{loss.item():.4f}")
            #目前进度条实时类似：Epoch 1/50: 5%|██ | 35/704 [00:01<00:02, loss=2.3041]


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

            checkpoint = {
                'epoch': epoch + 1, # 存下一轮
                'model_state_dict': net.state_dict(),
                ##推荐的方式是保存模型的 state_dict，这样可以更灵活地加载到不同架构中
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }

            #这里改了torch.save(net.state_dict(), best_model_path)，为一个字典，让保存更全面
            
            torch.save(checkpoint, best_model_path)
            print('目前最好的模型已保存，准确率为：{:.4f}'.format(best_val_acc))

            ###如果后面要加载，续训，应该这么写：
            ##加载字典
            #checkpoint = torch.load(best_model_path)
            ##恢复模型
            #net.load_state_dict(checkpoint['model_state_dict'])
            ##恢复优化器
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ##恢复学习率
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            ##从断掉的地方开始跑
            #start_epoch = checkpoint['epoch']

    print('-------------------------------训练循环结束，接下来是测试模型时间--------------------------------')

    checkpoint = torch.load(best_model_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    test_loss_acc,test_correct_acc=evaluate(net,test_loader,criterion,device)
    print(f'该模型的测试结果是:Test Loss: {test_loss_acc:.4f}, Test Acc: {test_correct_acc:.4f}')


    print('----------------------------------------测试结束------------------------------------------')


if __name__ == "__main__":
    main()

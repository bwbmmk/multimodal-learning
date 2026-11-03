import torch


class basicblock(torch.nn.Module):


    expansion=1
    #expansion 是残差块输出相对基础通道数的倍率。
    #BasicBlock 最终输出为 outchannels×1；输入输出通道仍可不同。ResNet-18/34 用 BasicBlock，ResNet-50 用 Bottleneck。

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


        self.shoutcut=torch.nn.Sequential()
        #nn.Sequential()是一个有序的容器，里面的模块会按照传入的顺序依次执行。
        if stride!=1 or inchannels!=outchannels:
            self.shoutcut=torch.nn.Sequential(
                torch.nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=stride,bias=False),
                torch.nn.BatchNorm2d(outchannels)
            )
            #残差相加要求两边形状一致，
            #1×1 二维卷积匹配通道和空间尺寸，同时学习输入通道的组合。


    def forward(self,x):
        identity=self.shoutcut(x)
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

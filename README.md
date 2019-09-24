# DRLPytorch-
《白话强化学习与PyTorch》的学习笔记

代码参考https://github.com/GAOYANGAU/DRLPytorch

# 第五章-时间差分
## 5.1 Qlearning.py
对源代码（native-Qlearning.py）结构进行了更改，主要体现在如下2个方面：
1. 随机产生初始点后，采用epsilon-贪婪发进行动作选择，当到达目标状态后为一个Episode
2. 动作价值更新公式为完整的Q计算公式（原代码将alpha取为了1）

# 第六章 深度学习
## 6.1 linear_regression.py 
对书中77页代码进行修改，使其适应0.4以后的版本，主要有一下3个部分
1. 无需再import torch.autograd.Variable，Tensor直接可以计算梯度
2. 累加损失时.data[0]改为.item()，0.4以后的版本中loss是一个零维的标量，用loss.item()可以从标量中获取Python数字。
3. 增加了对模型保存和加载，快速搭建神经网络的学习笔记
 
 
## 6.2 feedforward_neural_net.py
对书中119页代码修改为GPU版本，添加了GPUtil等几个小模块，实时监测GPU信息；
分别在CPU和GPU下运行后：CPU用时80s，GPU用时37s。

## 6.3 convoluntional_neural_network.py
对书中125页代码的学习过程中，将其修改为GPU版本后出现了如下错误:

![image](https://github.com/catziyan/DRLPytorch-/blob/master/erro.png)

考虑到在使用全连接网络时可以使用GPU，故放弃思考（繁琐的）cuda，cudnn的本版问题，直接使用torch.backends.cudnn.enabled = False 解决问题。
但具体为何造成此错误，作为小蚂蚁的我不得而知

与全连接神经网络相比，卷积神经网络的准确率更高（全连接97%、96%左右，卷积神经网络99%），但速度更慢，共用了105s，GPU的占用率也越高，用GPUtil模块打印结果如图：（之所以放图是因为刚学了如何在github插入图片，haha）

![image](https://github.com/catziyan/DRLPytorch-/blob/master/GPU.png)

## 6.4 Recurrent_Neural_net.py
对书中143页代码学习过程中，研究了两个细节问题：
1. 在计算正确率时，用（100*(correct/total)）计算得到结果的总是0。 因为correct是由tensor计算得到的，故correct也为tensor，且数据类型为torch.int64。 在pytorch中的int/long之间的运算得到的还是整形，故计算结果总为0.
2. 书中对out = self.fc(out[:,-1,:])分析为“这是一个二维张量，第一个维度是batch_size，第二个维度是input_size，尺寸为[100,28]”，而全连接网络的输入为elf.fc = torch.nn.Linear(hidden_size, output_size)，则显然out[:,-1,:]的尺寸应该为[batch_size, hidden_size]=[100,36].

## 橙大陈的第六章个人总结
第六章用了三种神经网络实现了对手写数字(0~9)数据集MNIST的分类问题（全连接神经网络（6.2）、卷积神经网络（6.3）、循环神经网络（6.4）），其中在学习过程中感觉比较绕的是各个网络的输入和输出，下面我再来梳理一下：
首先，需要明确从MNIST下载的图片为灰度图片，分辨率为28像素×28像素，其中train_dataset/test_dataset存在两种数据，一个是大小为[1，28，28]图片数据（data），一个是int类型的标签(targets)。用dataloader载入数据后，images.size为[batch_size,1,28,28],labels.size为[100].
1. 全连接神经网络参数：

Net(

&#160;&#160; (linear1): Linear(in_features=784, out_features=500, bias=True)
  
&#160;&#160;  (relu): ReLU()
  
&#160;&#160;  (linear2): Linear(in_features=500, out_features=10, bias=True)
  
)

故网络的输入为28×28，故需要对从dataloader取出的数据进行处理后输入（images.view(-1, 28×28)→images.size([batch_size,784])）

x = self.linear1(x)   → x.size([100, 500])

x = self.relu(x)      → x.size([100, 500])

out = self.linear2(x) → x.size([100, 10]) 

（输出为独热向量，适用于多分类问题，用交叉熵损失函数（CrossEntropyLoss）训练）


2. 卷积神经网络参数：

第一层网络：直接输入图片信息，大小为（batch_size,1,28,28），输出为大小为（batch_size, 16 ,14,14）

self.layer1 = nn.Sequential(
    
   nn.Conv2d(
   
             in_channels=1,   #图片高度(1)
              
             out_channels=16, #卷积核个数
              
             kernel_size=5,   #卷积核尺寸
              
             stride=1,        #卷积核扫描步长
              
             padding=2),      #边缘补零  
    
   nn.BatchNorm2d(16),        #BN层，归一化操作，output shape（batch_size, 16 ,28,28）
    
   nn.ReLU(),                 #output shape（batch_size, 16 ,28,28）
  
   nn.MaxPool2d(2))           #池化层,output shape（batch_size, 16 ,14,14）

第二层网络：输入大小为（batch_size,16,14,14），输出为大小为（batch_size, 32 ,7,7）

self.layer2 = nn.Sequential(   
   
&#160;&#160;&#160;   nn.Conv2d(16,32,5,1,2),    #卷积层，output shape（batch_size, 32 ,14,14）
    
&#160;&#160;&#160;   nn.BatchNorm2d(32),        #output shape（batch_size, 32 ,14,14）
    
&#160;&#160;&#160;   nn.ReLU(),
    
&#160;&#160;&#160;   nn.MaxPool2d(2)            #output shape（batch_size, 32 ,7,7）

)

第三层网络： 输入大小为（batch_size,7*7*32）,故需要对第二层网络的输出out进行处理（out.view(out.size(0), -1)），输出大小为（batch_size,10）
self.linear = nn.Linear(7*7*32, 10)

(除此之外，因为网络中用到了BN层，所以在测试的时候，需要进入评估模式（net.eval()），评估模式下的BN层的均值和方差为整个训练集的均值和方差，而训练模式下的BN层的均值和方差为batch_size的均值和方差)


3. 循环神经网络参数：

self.lstm = torch.nn.LSTM(         # if use nn.RNN(), it hardly learns
   
    input_size=input_size,
    
    hidden_size=32,         # rnn hidden unit
    
    num_layers=2,           # number of rnn layer
    
    batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
)

self.fc = torch.nn.Linear(hidden_size, output_size)

循环神经网络的输入为x为（batch_size, sequence_length/time_step(本例中为图片的长/行数), input_size（图片的宽/列数）），故需要对images进行处理images.view(-1, sequence_length, input_size)

out, (h0, c0) = self.lstm(x, None)

out shape（batch_size, sequence_length/time_step, hidden_size）#不知道为什么这本书和莫烦的代码里都写的r_out shape (batch, time_step, output_size)，本例我打印out之后的结果为torch.Size([100, 28, 32])！

h0 shape(num_layer, batch_size, hidden_size)

c0 shape(num_layer, batch_size, hidden_size)

只需取循环神经网络的最后一个time_step数据作为全连接层的输入，故对out进行处理out[:,-1,:],该处理使out shape(100,28,32) →（100，32）输入全连接层



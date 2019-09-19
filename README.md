# DRLPytorch-
《白话强化学习与PyTorch》的学习笔记

代码参考https://github.com/GAOYANGAU/DRLPytorch


## Qlearning.py
对源代码（native-Qlearning.py）结构进行了更改，主要体现在如下2个方面：
1. 随机产生初始点后，采用epsilon-贪婪发进行动作选择，当到达目标状态后为一个Episode
2. 动作价值更新公式为完整的Q计算公式（原代码将alpha取为了1）


## linear_regression.py 
对书中77页代码进行修改，使其适应0.4以后的版本，主要有一下3个部分
1. 无需再import torch.autograd.Variable，Tensor直接可以计算梯度
2. 累加损失时.data[0]改为.item()，0.4以后的版本中loss是一个零维的标量，用loss.item()可以从标量中获取Python数字。
3. 增加了对模型保存和加载，快速搭建神经网络的学习笔记
 
 
## feedforward_neural_net.py
对书中125页代码修改为GPU版本，添加了GPUtil等几个小模块，实时监测GPU信息；
分别在CPU和GPU下运行后：CPU用时80s，GPU用时37s。

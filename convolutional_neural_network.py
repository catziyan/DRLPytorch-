#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:CatZiyan
# @Time :2019/9/20 10:35

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import time
import GPUtil
import os

start = time.clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = False
# hyper parameters
batch_size = 100
learning_rate = 0.001
Epoch = 5
if not (os.path.exists('./data/')) or not (os.listdir('./data/')):
    DOWNLOAD = True
else:
    DOWNLOAD = False


# load data
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=DOWNLOAD)
test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size =batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.linear = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out

net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# train
for epoch in range(Epoch):
    GPUtil.showUtilization()
    for i,(images, labels) in enumerate(train_loader):
        predicted = net(images.to(device))
        loss = criterion(predicted, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1)%100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  %(epoch+1, Epoch, i+1, len(train_dataset)//batch_size, loss.item()))



# test
net = net.eval() #进入评估模式（评估模式下的BN层的均值和方差为整个训练集的均值和方差，而训练模式下的BN层的均值和方差为mini-batch的均值和方差）
correct = 0
total = 0
for (images, labels) in test_loader:
    predicted = net(images.to(device))
    _, predicted = torch.max(predicted.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.to(device)).sum()

print('Test Accuracy of the model on the %d test images: %d %%' %(total, 100*correct/total))

torch.save(net.state_dict(), 'model.pkl')

net2 = CNN().to(device)
net2.load_state_dict(torch.load('model.pkl'))

elapsed = time.clock() - start
print('Time used:', elapsed)

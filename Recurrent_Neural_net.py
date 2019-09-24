#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author:CatZiyan
# @Time :2019/9/24 9:36

import torch
import torchvision
import time
import os
import GPUtil

start = time.clock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

# hyper parameters
batch_size = 100
input_size = 28
sequence_length = 28
hidden_size = 32
output_size = 10
num_layers = 2
Epoch = 5
learning_rate = 0.001
if not (os.path.exists('./data/')) or not (os.listdir('./data/')):
    DOWNLOAD = True
else:
    DOWNLOAD = False

# download data
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=DOWNLOAD)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor())

# load dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# RNN
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(device)
        # out,_ = self.lstm(x,(h0,c0))

        out, _ = self.lstm(x, None)
        out = self.fc(out[:,-1,:])
        return out

net = RNN(input_size, hidden_size, num_layers, output_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
#
# train net
for epoch in range(Epoch):
    GPUtil.showUtilization()
    for i,(images, labels) in enumerate(train_loader):
        images = images.to(device).view(-1, sequence_length, input_size)
        labels = labels.to(device)
        predicted = net(images)
        optimizer.zero_grad()
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [%d/%d], step [%d/%d], loss: %.4f' %(epoch+1, Epoch, i+1, len(train_dataset)//batch_size, loss.item()))
torch.save(net.state_dict(), 'rnn.pkl')
# test data

net2 = RNN(input_size, hidden_size, num_layers, output_size).to(device)
net2.load_state_dict(torch.load('rnn.pkl'))
total = 0
correct = 0
for images,labels in test_loader:
    images = images.to(device).view(-1, sequence_length, input_size)
    labels = labels.to(device)
    predicted = net2(images)
    _, predicted = torch.max(predicted.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
#     correct += (predicted == labels).sum().float()
# print(type(total),type(correct))
# print(correct.dtype)
# print('Test Accuracy of the model on the %d test images:%d%%' %(total, 100*(correct/total)))
print('Test Accuracy of the model on the %d test images: %d %%' %(total, 100*correct/total))


elapsed = time.clock()-start
print('Time used:', elapsed)

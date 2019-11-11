import torch
from torch import nn


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 16, kernel_size=8, stride=4),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.MaxPool2d(2),

        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

class DuelingNet(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(DuelingNet, self).__init__()
        self.input_shape = inputs_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 16, kernel_size=8, stride=4),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=4, stride=2),
            nn.MaxPool2d(2)
        )
        self.hidden = nn.Sequential(
            nn.Linear(self.features_size(), 256, bias=True),
            nn.ReLU()
        )

        self.adv = nn.Linear(256, num_actions, bias=True)
        self.val = nn.Linear(256, 1, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        adv = self.adv(x)
        val = self.val(x).expand(x.size(0),self.num_actions) #扩展某个size为1的维度，值一样  （1，6）
        x = val + adv -adv.mean(1).unsqueeze(1).expand(x.size(0),self.num_actions)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
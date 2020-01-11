#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author:CatZiyan
# @Time :2020/1/9 19:43
import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#Hyperparameters
RL_A = 0.001
RL_C =0.01
gamma = 0.9
num_episode = 10000
ENVNAME = 'CartPole-v0'
env = gym.make(ENVNAME)
env = env.unwrapped
state_space = env.observation_space.shape[0]
action_space = env.action_space.n #2
env.seed(1)  # reproducible

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = nn.Linear(self.state_space, 32)
        self.fc2 = nn.Linear(32, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x

class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_space,32)
        self.out = nn.Linear(32,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return  x

class PG(object):
    def __init__(self):
        self.policy = Policy()
        self.Q_net = Q_net()
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(),lr=RL_A)
        self.optimizer_q = torch.optim.Adam(self.Q_net.parameters(),lr=RL_C)
        self.loss_td = nn.MSELoss()

    def choose_action(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        probs = self.policy(state)
        c = Categorical(probs)
        action =c.sample()
        return  int(action.data.numpy())

    def learn(self,state,action,reward,next_state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        next_state = torch.unsqueeze(torch.FloatTensor(next_state), 0)
        reward = torch.FloatTensor([reward])
        v = self.Q_net(state)
        v_ = self.Q_net(next_state)
        td_error = reward + v_ *gamma -v
        loss_q = self.loss_td(reward + v_ *gamma,v)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        action = torch.FloatTensor([action])
        probs = self.policy(state)
        c =  Categorical(probs)
        loss = -c.log_prob(action) * td_error.item()
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()


pg = PG()
episode_durations = []
reward_total = 0
steps = 0
for episode in range(num_episode):
    state = env.reset()
    reward_total = 0
    for t in range(10000):
        action = pg.choose_action(state)
        next_state, reward, done, info = env.step(action)
        # env.render()
        pg.learn(state,action,reward,next_state)
        state = next_state
        state = torch.from_numpy(state).float()
        steps += 1
        reward_total = reward_total+reward
        if done:
            episode_durations.append(t + 1)

            steps = 0
            break

    print('episode:%d; reward:%d' % (episode, reward_total))
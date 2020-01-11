#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author:CatZiyan
# @Time :2020/1/7 21:45

import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.001
gamma = 0.99
num_episode = 1000
ENVNAME = 'CartPole-v0'
env = gym.make(ENVNAME)
env = env.unwrapped
state_space = env.observation_space.shape[0]
action_space = env.action_space.n #2

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class PG(object):
    def __init__(self):
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=learning_rate)
        self.state_pool = []
        self.action_pool= []
        self.reward_pool = []

    def choose_action(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        probs = self.policy(state)
        c = Categorical(probs)
        action =c.sample()
        return  int(action.data.numpy())

    def store_transition(self,state,action,reward):
        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)

    def learn(self):
        r = 0
        for i in reversed(range(len(self.reward_pool))):
            r = r*gamma+self.reward_pool[i]
            self.reward_pool[i] = r

        reward_mean = np.mean(self.reward_pool)
        reward_std = np.std(self.reward_pool)
        self.reward_pool = (self.reward_pool - reward_mean) / reward_std

        self.optimizer.zero_grad()
        for i in range(len(self.state_pool)):
            state = torch.unsqueeze(torch.FloatTensor(self.state_pool[i]), 0)
            action = torch.FloatTensor([self.action_pool[i]])
            reward = self.reward_pool[i]
            probs = self.policy(state)
            c =  Categorical(probs)
            loss = -c.log_prob(action) * reward
            # print(loss)
            loss.backward()

        self.optimizer.step()
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []

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
        pg.store_transition(state,action,reward)
        state = next_state
        state = torch.from_numpy(state).float()
        steps += 1
        reward_total = reward_total+reward
        if done:
            episode_durations.append(t + 1)
            pg.learn()
            steps = 0
            break

    print('episode:%d; reward:%d' % (episode, reward_total))

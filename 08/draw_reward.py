#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author:CatZiyan
# @Time :2019/10/22 12:00
import numpy as np
import matplotlib.pyplot as plt

def Reward(dir):
    f = open(dir, 'r')
    lines = f.readlines()  # 返回所有的行
    lines = lines[0:1000]
    reward = []
    reward_mean = []
    for i in range(len(lines)):
        line = lines[i]
        data = line.split()  #reward 与/n 分割
        reward.append((float(data[0])))
        reward_mean.append(np.mean(reward[-100:]))

    f.close()
    return reward, reward_mean



dir = './model/lr0.0001-21/rewardfs.txt'
dir2 = './model/lr0.0001-1/reward.txt'
dir3 = './model/ddqn-lr0.0001-1/double_dqn_reward.txt'
dir4 = './model/double_dqn_reward.txt'

reward, reward_mean = Reward(dir4)
reward2, reward_mean2 = Reward(dir2)
reward3, reward_mean3 = Reward(dir4)

plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(range(len(reward)),reward,label='nature_dqn,|reward| = 21,done = Ture')
# plt.plot(range(len(reward2)),reward2,label='nature_dqn,|reward| = 1, done = Ture')
# plt.plot(range(len(reward3)),reward3,label='double_dqn,|reward| = 1, done = Ture')


# plt.legend()

plt.subplot(122)
plt.plot(range(len(reward_mean)),reward_mean,label='nature_dqn,|reward| = 21,done = Ture')
# plt.plot(range(len(reward_mean2)),reward_mean2,label='nature_dqn,|reward| = 1, done = Ture')
# plt.plot(range(len(reward_mean3)),reward_mean3,label='double_dqn,|reward| = 1, done = Ture')
# plt.legend()
plt.show()
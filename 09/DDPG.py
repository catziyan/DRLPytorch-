# '''
# torch = 0.41
# '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
ENV_NAME = 'Pendulum-v0'

hidden_size = 64

###############################  DDPG  ####################################

class ANet(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,hidden_size)

        self.out = nn.Linear(hidden_size,a_dim)

    def forward(self,x):
        x = self.fc1(x)
        torch.nn.Dropout(0.5)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)     #输出范围为-1,1
        actions_value = x*2    #更改输出范围print(env.action_space.high)
        return actions_value

class CNet(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim,hidden_size)
        self.fca = nn.Linear(a_dim,hidden_size)
        self.out = nn.Linear(hidden_size,1)

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        torch.nn.Dropout(0.5)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s,test):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if test == True:
            self.Actor_eval.eval()
        else:
            self.Actor_eval.train()
        return self.Actor_eval(s)[0].detach() # ae（s）

    def learn(self):
        self.Actor_eval.train()
        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - TAU) + param.data * TAU
            )
        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - TAU) + param.data * TAU
            )
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])
        #更新当前Actor网络
        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        #更新当前Critic网络
        a_ = self.Actor_target(bs_)
        q_ = self.Critic_target(bs_,a_)
        q_target = br+GAMMA*q_
        q_v = self.Critic_eval(bs,ba)
        td_error = self.loss_td(q_target,q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
# env.seed(1)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for episode in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    done = False
    for j in range(MAX_EP_STEPS):
        # env.render()
        # Add exploration noise
        a = ddpg.choose_action(s,False)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r , s_)
        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            f = open('./model/train_fit.txt', 'a+')
            f.write(str(int(ep_reward)))
            f.write('\n')
            f.close()
            # if ep_reward > -300:RENDER = True
            break
    if episode % 10 == 0:
      total_reward = 0
      for i in range(10):
        state = env.reset()
        for j in range(MAX_EP_STEPS):
          # env.render()
          action = ddpg.choose_action(state,True) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/10
      print ('episode: ',episode,'Evaluation Average Reward:',float(ave_reward))
      f = open('./model/test_fit.txt', 'a+')
      f.write(str(float(ave_reward)))
      f.write('\n')
      f.close()

print('Running time: ', time.time() - t1)



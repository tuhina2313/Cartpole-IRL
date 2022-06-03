# -*- coding: utf-8 -*-
import os
import gym
import math
import copy
import torch
import tqdm
import random
import pathlib
import argparse
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use("TkAgg")
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from itertools import count
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from collections import namedtuple

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
       
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

HIDDEN_LAYER = 64  # NN hidden layer size
class DQN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l1_1 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l1_2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l1_1(x))
        x = F.relu(self.l1_2(x))
        x = self.l2(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Trainer(object):

    device = torch.device("cpu") 

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.95
    num_episodes = 200

    EPS_END = 0.05
    EPS_DECAY = num_episodes * 0.9
    TARGET_UPDATE = 10
    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def __init__(self, env, name):

        self.env = env
        self.env.reset()
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.is_trained = False
        self.avgFeature = None

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.best_model = None
        self.best_rwd = -float('inf')

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(100000)

        self.NUM_UPDATE = 1
        self.steps_done = 0
        self.episode_durations = []
        self.name = name

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        for i in range(self.NUM_UPDATE):
            transitions = self.memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def testModel(self, mdl, save_states=False):
        ep_rwd = 0
        state_list = []
        state_tp = self.env.reset()
        state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.float)
        if save_states:
            state_list.append(self.featurefn(state_tp))
        with torch.no_grad():
            for t in count():
                a = self.policy_net(state).max(1)[1].view(1, 1)
                state_tp, reward, done, _ = self.env.step(a.item())
                state = torch.from_numpy(state_tp).unsqueeze(0).to(self.device, dtype=torch.float)
                if save_states:
                    state_list.append(self.featurefn(state_tp))
                ep_rwd += reward
                if done or t > 30000:
                    break

        if ep_rwd > self.best_rwd and not save_states:
            self.best_rwd = ep_rwd
            self.best_model = copy.deepcopy(mdl)
        if not save_states:
            return ep_rwd
        else:
            return ep_rwd, state_list

    def featurefn(self, state):

        x, x_dot, theta, theta_dot = state
        x = (x + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        
        x_dot = (x_dot + self.env.unwrapped.x_threshold) / (2 * self.env.unwrapped.x_threshold)
        theta = (theta + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
        theta_dot = (theta_dot + self.env.unwrapped.theta_threshold_radians) / (2 * self.env.unwrapped.theta_threshold_radians)
        feat = torch.tensor(
            [
                x, x_dot, theta, theta_dot,
                x ** 2, x_dot ** 2, theta ** 2, theta_dot ** 2,
            ]
        )
        return feat

    def train(self, rwd_weight=None):

        for i_episode in tqdm.tqdm(range(self.num_episodes)):

            state = torch.from_numpy(self.env.reset()).unsqueeze(0).to(self.device, dtype=torch.float)
            for t in count():

                action = self.select_action(state)
                next_state_np, reward, done, _ = self.env.step(action.item())
   
                #next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(self.device, dtype=torch.float)
                if rwd_weight is None:
                    reward = torch.tensor([reward], device=self.device)
                    x, x_dot, theta, theta_dot = next_state_np
                    r1 = (self.env.unwrapped.x_threshold - abs(x)) / self.env.unwrapped.x_threshold - 0.8
                    r2 = (self.env.unwrapped.theta_threshold_radians - abs(theta)) / self.env.unwrapped.theta_threshold_radians - 0.5

                    reward = torch.tensor([r1 + r2])
                else:
                    feat = self.featurefn(next_state_np)
                    reward = rwd_weight.t() @ feat

                if done:
                    next_state = None

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()
                if done or t > 30000:
                    self.episode_durations.append(t + 1)
                    self.showProgress(i_episode)
                    break

            policy_rwd = 0
            policy_rwd = self.testModel(self.policy_net)
            print('Policy Reward: %d' % policy_rwd)

            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        #
        # Done training.
        print('Complete')
        self.is_trained = True

    def showProgress(self, e_num):
        means = 0
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if len(self.episode_durations) >= 100:
            means = durations_t[-100:-1].mean().item()
        print('Episode %d/%d Duration: %d AVG: %d'%(e_num, self.num_episodes, durations_t[-1], means))
        plt.figure(2)
        plt.clf()
        plt.title('Performance: %s' % self.name)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

            plt.pause(0.001) 

    def saveBestModel(self):
        pathlib.Path('mdls/').mkdir(parents=True, exist_ok=True)
        state = {
            'mdl': self.best_model.state_dict(),
            'avgFeat': self.avgFeature
        }
        import datetime
        now = datetime.datetime.now()
        save_name = 'mdls/' + 'mdl_DATE-' + now.isoformat() + '.pth.tar'
        torch.save(state, save_name)

    def gatherAverageFeature(self):
        with torch.no_grad():
            n_iter = 2000
            sample_sum = None
            rwd_sum = None
            for i in tqdm.tqdm(range(n_iter)):
                rwd, states = self.testModel(self.best_model, True)
                episodeMean = torch.stack(states).mean(0)
                if sample_sum is None:
                    sample_sum = episodeMean
                    rwd_sum = rwd
                else:
                    sample_sum += episodeMean
                    rwd_sum += rwd
            sample_sum /= n_iter
            rwd_sum /= n_iter
            print(sample_sum)
            print(rwd_sum)
        self.avgFeature = sample_sum
        return sample_sum, rwd_sum

class inverse_RL(object):

    def __init__(self, env):
        self.env = env
        self.expert = DQN_Trainer(self.env, 'Expert')
        if not self.expert.is_trained:
            self.expert.train()
            self.expert.gatherAverageFeature()
            self.expert.saveBestModel()

        if self.expert.avgFeature is None:
            self.expert.gatherAverageFeature()
            state = {
                'mdl': self.expert.policy_net.state_dict(),
                'avgFeat': self.expert.avgFeature
            }
            torch.save(state)
        self.expert_feat = self.expert.avgFeature

    def train(self):
        robot = DQN_Trainer(self.env, 'Robot')
        sampleFeat = robot.featurefn(self.env.reset())
        w_0 = torch.rand(sampleFeat.size(0), 1)
        w_0 /= w_0.norm(1)
        rwd_list = []
        t_list = []
        weights = [w_0]
        i = 1

        robot.train(w_0)
        robotFeat, robotRwd = robot.gatherAverageFeature()
        rwd_list.append(robotRwd)
        t_list.append((self.expert_feat - robotFeat).norm().item())

        weights.append((self.expert_feat - robotFeat).view(-1, 1))
        feature_bar_list = [robotFeat]
        feature_list = [robotFeat]

        n_iter = 20
        for i in tqdm.tqdm(range(n_iter)):
            robot = DQN_Trainer(self.env, 'robot_%d' % (i + 1))
            robot.train(weights[-1])
            robotFeat, robotRwd = robot.gatherAverageFeature()
            rwd_list.append(robotRwd)
            feature_list.append(robotFeat)
            feat_bar_next = feature_bar_list[-1] + ((feature_list[-1] - feature_bar_list[-1]).view(-1, 1).t() @ (self.expert_feat - feature_bar_list[-1]).view(-1,1))\
                             / ((feature_list[-1] - feature_bar_list[-1]).view(-1, 1).t() @ (feature_list[-1] - feature_bar_list[-1]).view(-1,1))\
                             * (feature_list[-1] - feature_bar_list[-1])
            feature_bar_list.append(feat_bar_next)
            weights.append((self.expert_feat - feat_bar_next).view(-1, 1))
            t_list.append((self.expert_feat - feat_bar_next).norm().item())
            print('t: ', t_list[-1])
        print(feat_bar_next)
        plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(rwd_list)
        plt.title('Average Episode Reward')
        plt.xlabel('robot Number')
        plt.ylabel('Episode Length')
        plt.savefig('plts/avgRewardProgress.png')
        plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(t_list)
        plt.title('L2 Policy Error')
        plt.xlabel('robot Number')
        plt.ylabel('Squared error of features of features')
        plt.savefig('plts/sqerr.png')

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    arl = inverse_RL(env)
    arl.train()

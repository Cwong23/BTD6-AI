import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy
import gym
from gym import spaces

from actions import *
from takeScreenShot import *

import numpy as np

class BTDEnv(gym.Env):
    def __init__(self):
        super(BTDEnv, self).__init__()
        # define observation and action space
        self.action_type_space = spaces.MultiDiscrete([3, 1550, 920, 2, 100, 3]) # action, x, y, monkey, index, upgrade path
        self.state = None
        self.rounds = 0
        self.health = 0
        self.money = 0

    def reset(self):
        # returns the observation of the intial state
        # reset the environment
        # round, money, previous moves
        self.rounds = int(scRound()[0])
        self.health = int(scLives())
        self.money = int(scMoney())
        resetGame()
        return self.state

    def step(self, action):
        if self.check_done():
            return False
        self.rounds+=1
        action_type = action[0] # What type of action
        # For buy
        x = 50+action[1] # x coord
        y = 80+action[2] # y coord
        m = action[3] # monkey chosen to buy
        # For upgrade
        t = action[4] # used to get an index position of the current monkies
        u = action[5] # used to select which upgrade to buy
        
        if action_type == 0:
            buy(x, y, monkey_dict[m])
        elif action_type == 1:
            upgrade(current_monkeys[t%len(current_monkeys)], u)
        else:
            pass

        reward = self.calculate_reward()
        done = self.check_done()
        return self.state, reward, done

    def calculate_reward(self):
        nHealth = self.health / 150 # the 150 is max health
        nRounds = self.rounds / 100 # ditto
        reward = nHealth + nRounds

        if self.rounds < 90: # positive reward for more money in early rounds, negative reward for more money in later rounds
            reward += self.money/10000
        else:
            reward -= self.money/10000

        return reward

    def check_done(self):
        if not scDef():
            return False
        return True

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
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
import gymnasium as gym
from gymnasium import spaces

from actions import *
from takeScreenShot import *

import numpy as np

class BTDEnv(gym.Env):
    def __init__(self):
        super(BTDEnv, self).__init__()
        # define observation space
        self.observation_space = spaces.Box(
            low=np.zeros(103, dtype=np.float32),  
            high=np.ones(103, dtype=np.float32),
            shape=(3+100,),
            dtype=np.float32
        )
        # define action space
        self.action_space = spaces.MultiDiscrete([3, 1550, 920, 2, 100, 3]) # action, x, y, monkey, index, upgrade path
        self.state = None

        # define other variables
        self.rounds = 0
        self.health = 0
        self.money = 0

    def reset(self, seed=None, options=None):
        # reset the environment
        resetGame()
        self.done = False
        self.rounds = int(scRound()[0])
        self.health = int(scLives())
        self.money = int(scMoney())
        nHealth = max(0, min(1, self.health / 150))  # value is between 0 and 1
        nRounds = max(0, min(1, self.rounds / 100))  # value is between 0 and 1
        log_money = np.log1p(max(0, self.money))  # money is non-negative
        nMoney = max(0, min(1, log_money / 10000))  # value is between 0 and 1
        self.prev_actions = deque(maxlen=100) # max rounds
        for _ in range(100):
            self.prev_actions.append(0)

        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32)
        return self.observation, {}

    def step(self, action):
        self.prev_actions.append(action)
        if self.check_done():
            self.done = True
            truncated = False 
            return self.observation, self.calculate_reward(), self.done, truncated, {}

        self.rounds += 1
        action_type = action[0]  # What type of action
        # For buy
        x = 50 + action[1]  # x coord
        y = 80 + action[2]  # y coord
        m = action[3]  # Monkey chosen to buy
        # For upgrade
        t = action[4]  # Used to get an index position of the current monkeys
        u = action[5]  # Used to select which upgrade to buy
        
        if action_type == 1:
            buy(x, y, monkey_dict[m])
        elif action_type == 2:
            upgrade(current_monkeys[t % len(current_monkeys)], u)
        else:
            pass
        
        self.rounds = int(scRound()[0])
        self.health = int(scLives())
        self.money = int(scMoney())
        nHealth = max(0, min(1, self.health / 150))  # value is between 0 and 1
        nRounds = max(0, min(1, self.rounds / 100))  # value is between 0 and 1
        log_money = np.log1p(max(0, self.money))  # money is non-negative
        nMoney = max(0, min(1, log_money / 10000))  # value is between 0 and 1
        
        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32)
        self.reward = self.calculate_reward()
        self.done = self.check_done()
        truncated = False
        return self.observation, self.reward, self.done, truncated, {}

    def calculate_reward(self):
        nHealth = max(0, min(1, self.health / 150))  # value is between 0 and 1
        nRounds = max(0, min(1, self.rounds / 100))  # value is between 0 and 1
        reward = nHealth + nRounds
        log_money = np.log1p(max(0, self.money))  # money is non-negative
        nMoney = max(0, min(1, log_money / 10000))  # value is between 0 and 1
        if self.rounds < 90:  # Positive reward for more money in early rounds, negative reward for more money in later rounds
            reward += nMoney
        else:
            reward -= nMoney

        reward = np.clip(reward, -np.inf, np.inf)

        return reward

    def check_done(self):
        return self.health <= 0 or self.rounds >= 100

    def seed(self, seed=None):
        np.random.seed(seed)
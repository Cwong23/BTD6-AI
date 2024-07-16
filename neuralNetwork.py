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
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from actions import *
from takeScreenShot import *

class BTDEnv(gym.Env):
    def __init__(self):
        super(BTDEnv, self).__init__()
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.zeros(103, dtype=np.float32),  
            high=np.ones(103, dtype=np.float32),
            shape=(103,),
            dtype=np.float32
        )
        # Define action space
        self.action_space = spaces.MultiDiscrete([3, 1550, 920, 2, 100, 3]) # action, x, y, monkey, index, upgrade path
        self.state = None
        self.rounds = 0
        self.health = 0
        self.money = 0

    def reset(self, seed=None, options=None):
        # Reset the environment
        resetGame()
        self.done = False
        self.rounds = int(scRound()[0])
        self.health = int(scLives())
        self.money = int(scMoney())
        nHealth = max(0, min(1, self.health / 150))  # Ensure value is between 0 and 1
        nRounds = max(0, min(1, self.rounds / 100))  # Ensure value is between 0 and 1
        log_money = np.log1p(abs(self.money))
        nMoney = max(0, min(1, log_money / 10000))  # Ensure value is between 0 and 1
        self.prev_actions = deque([0] * 100, maxlen=100)  # Initialize with zeros to ensure correct length

        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32)
        return self.observation, {}

    def step(self, action):
        pyautogui.sleep(0.5)
        self.prev_actions.append(min(max(action[0] / 2.0, 0.0), 1.0))
        print(action)
        if scCurrent():
            restart()
            self.done = True
            return self.observation, 0, self.done, False, {}
        
        self.rounds += 1
        action_type = action[0]  # What type of action
        # For buy
        x = 50 + action[1]  # x coord
        y = 80 + action[2]  # y coord
        m = action[3]  # monkey chosen to buy
        # For upgrade
        t = action[4]  # used to get an index position of the current monkeys
        u = action[5]  # used to select which upgrade to buy
        
        if action_type == 1:
            buy(x, y, monkey_dict[m])
            pyautogui.sleep(0.5)
        elif action_type == 2:
            if len(current_monkeys) != 0:
                upgrade(current_monkeys[t % len(current_monkeys)], u)
            pyautogui.sleep(0.5)
        else:
            pass

        startRound()
        
        
        while scCurrent():
            pyautogui.sleep(3)  # Adjust the sleep time as needed
        
        self.rounds = int(scRound()[0])
        self.health = int(scLives())
        self.money = int(scMoney())
        nHealth = max(0.0, min(1.0, self.health / 150.0))  # Ensure value is between 0 and 1
        nRounds = max(0.0, min(1.0, self.rounds / 100.0))  # Ensure value is between 0 and 1
        log_money = np.log1p(abs(self.money))
        nMoney = max(0.0, min(1.0, log_money / 10.0))  # Adjust normalization if needed

        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32)
        self.reward = self.calculate_reward()
        self.done = self.check_done()
        truncated = False  # This can be set based on additional logic if needed
        return self.observation, self.reward, self.done, truncated, self.rounds

    def calculate_reward(self):
        nHealth = self.health / 150  # the 150 is max health
        nRounds = self.rounds / 100  # ditto
        reward = nHealth + nRounds
        log_money = np.log1p(abs(self.money))
        nMoney = log_money / 10000
        if self.rounds < 90:  # positive reward for more money in early rounds, negative reward for more money in later rounds
            reward += nMoney
        else:
            reward -= nMoney

        return reward

    def check_done(self):
        return scDef()

    def seed(self, seed=None):
        np.random.seed(seed)

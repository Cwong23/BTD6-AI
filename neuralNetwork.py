import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

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
        self.observation = None 

    def reset(self, seed=None, options=None):
        # Reset the environment
        resetGame()
        self.done = False
        pyautogui.sleep(0.5)
        self.rounds = scRound()[0]
        self.rounds = int(self.rounds)
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
        print("Rounds: ", self.rounds)
        ''' if scCurrent():
            restart()
            self.done = True
            return self.observation, 0, self.done, False, {}'''
        
        
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
        
        if not scFast():
            startRound()

        while scCurrent():
            pyautogui.sleep(3)  # Adjust the sleep time as needed
            if scDef():
                break
        
        
        self.health = int(scLives())
        self.money = int(scMoney())
        print("Lives: ", self.health)
        print("Money: ", self.money)
        self.rounds += 1
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
        '''
        log_money = np.log1p(abs(self.money))
        nMoney = log_money / 10000
        if self.rounds < 90:  # positive reward for more money in early rounds, negative reward for more money in later rounds
            reward += nMoney
        else:
            reward -= nMoney
        '''
        return reward

    def check_done(self):
        return scDef()

    def seed(self, seed=None):
        np.random.seed(seed)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  
        self.linear2 = nn.Linear(hidden_size, hidden_size)  
        self.linear3 = nn.Linear(hidden_size, output_size)  

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr 
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)  # Keep as float for easier indexing
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Ensure action is used correctly for indexing
            action_idx = action[idx].long()  # Convert to long for indexing
            if len(action_idx) == target.size(1):
                target[idx, :] = Q_new
            else:
                for i in range(len(action_idx)):
                    if action_idx[i] < target.size(1):
                        target[idx, action_idx[i]] = Q_new
                    else:
                        raise IndexError(f"Action index {action_idx[i]} is out of bounds for dimension 1 with size {target.size(1)}")

        # 2: r + y * max next_predicted Q value
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

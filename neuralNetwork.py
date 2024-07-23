import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from actions import *
from takeScreenShot import *

'''
Notes: The code is modified from Sentdex's and freeCodeCamp's base code.
'''

'''
BTDEnv(gym.Env)
Purpose: Custom env to have Agent interact with BTD 6
Inputs: gym.ENV
'''

'''
Notes:
    Action Space: Has a dimension of 6
        action (0-2), x(0-1550), y(0-920), monkey(0-2) BASED ON THE MONKEY DICT, index(0-100), upgrade path (0-2)
    Observation space: Has a dimension of 103 based on the 3 different variables round, health, and money, and another 100
                       spaces for the previous actions. It has a low of 0 and a high of 1
'''
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
        self.prev_health = 0
        self.observation = None 

    '''
    reset(BTDEnv, None, None) -> tuple
    Purpose: Resets the enviornment after a loss
    Inputs: None
    Outputs: tuple
    Logic: Restarts game. Updates the new variables. Normalizes the values so that it is between 0 and 1 so it can fit in the observation space.
    '''

    def reset(self, seed=None, options=None):
        # Reset the environment
        resetGame()
        self.done = False
        pyautogui.sleep(0.5)
        self.rounds = scRound()[0]
        self.rounds = int(self.rounds)
        self.health = int(scLives())
        self.prev_health = self.health
        self.money = int(scMoney())
        nHealth = max(0, min(1, self.health / 200))  # Ensure value is between 0 and 1
        nRounds = max(0, min(1, self.rounds / 40))  # Ensure value is between 0 and 1
        log_money = np.log1p(abs(self.money))
        nMoney = max(0, min(1, log_money / 10000))  # Ensure value is between 0 and 1
        self.prev_actions = deque([0] * 100, maxlen=100)  # Initialize with zeros to ensure correct length

        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32)
        return self.observation, {}
    
    '''
    step(BTDEnv, list) -> tuple
    Purpose: Used to interact with the game and train the nn
    Inputs: list
    Outputs: tuple
    Logic: Performs action and waits until round is over. Then calculates a reward and creates an observation.
    '''

    def step(self, action):
        pyautogui.sleep(0.5)
        self.prev_actions.append(min(max(action[0] / 2.0, 0.0), 1.0))
        print(action)
        print("Rounds: ", self.rounds)
        
        action_type = action[0]  # What type of action
        # For buy
        x = 50 + action[1]  # x coord
        y = 80 + action[2]  # y coord
        m = action[3]  # monkey chosen to buy
        # For upgrade
        t = action[4]  # used to get an index position of the current monkeys
        u = action[5]  # used to select which upgrade to buy
        
        if action_type == 1: # buy
            buy(x, y, monkey_dict[m])
            pyautogui.sleep(0.5)
        elif action_type == 2: # upgrade
            if len(current_monkeys) != 0:
                upgrade(current_monkeys[t % len(current_monkeys)], u)
            pyautogui.sleep(0.5)
        else: # other
            pass
        
        startRound() # starts round
        
        if not scFast(): # checks if it is fast forwarded
            startRound()

        while scCurrent(): # waits until round is over
            pyautogui.sleep(3)  
            if scDef(): # checks for defeat
                break
        
        # updates variabels
        self.health = int(scLives()) 
        self.money = int(scMoney())
        print("Lives: ", self.health)
        print("Money: ", self.money)
        self.rounds += 1
        nHealth = max(0.0, min(1.0, self.health / 200.0))  # value is between 0 and 1
        nRounds = max(0.0, min(1.0, self.rounds / 40.0))  # value is between 0 and 1
        log_money = np.log1p(abs(self.money))
        nMoney = max(0.0, min(1.0, log_money / 10000.0))  # normalization if needed

        self.observation = np.array([nHealth, nRounds, nMoney] + list(self.prev_actions), dtype=np.float32) # creates an observation
        self.reward = self.calculate_reward() # reward calculation
        self.done = self.check_done() # checks if the game is done
        truncated = False
        
        return self.observation, self.reward, self.done, truncated, self.rounds

    '''
    calculate_reward() -> float
    Purpose: Calculate a reward based on the different variables
    Input: None
    Output: float
    Logic: Normalizes the values and adds them
    '''

    def calculate_reward(self) -> float:
        nHealth = (self.prev_health - self.health) / 200  # the 200 is max health for easy
        self.prev_health = self.health
        nRounds = self.rounds / 40  # ditto
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

    # checks if it is defeated or not
    def check_done(self):
        return scDef()

    # used for randomness
    def seed(self, seed=None):
        np.random.seed(seed)

'''
Linear_QNet(nn.Module)
Purpose: Neural Network that is connected with one hidden layer.
Inputs: nn.Module
'''

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # transformation from input size to the hidden size
        self.linear2 = nn.Linear(hidden_size, hidden_size)  # transformation of a hidden layer
        self.linear3 = nn.Linear(hidden_size, output_size)  # transformation from hidden size to output size

    '''
    forward(self, x) -> list
    Purpose: Applies linear transformations defined in the init
    Inputs: self, list
    Outputs: list
    '''

    def forward(self, x):
        x = F.relu(self.linear1(x)) # applies a linear transformation with a ReLU activation 
        x = F.relu(self.linear2(x)) # ditto
        x = self.linear3(x) # final linear transformation
        return x
    
    '''
    save(self, file_name) -> None
    Purpose: Saves the model's parameters to a file
    Inputs: self, str
    Outputs: None
    '''

    def save(self, file_name = 'model.pth'):
        model_folder_path = './model' # folder to be saved
        if not os.path.exists(model_folder_path): # creates folder if it doesn't exist
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name) # constructs save path
        torch.save(self.state_dict(), file_name) # saves the model's state dictionary to the file

'''
QTrainer
Purpose: Trainer for Q-learning model
Notes: 
    Loss function is mean error squared
'''

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr # learning rate
        self.gamma = gamma # discount factor
        self.model = model # model to be trained
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr) # optimizer for updating parameters
        self.criterion = nn.MSELoss() # Loss function to mean squared error

    '''
    train_step(self, list, list, int, list, bool) -> None
    Purpose: Converts inputs to PyTorch tensors to ensure they are the correct shape
    Inputs: self, list, list, int, list, bool
    Outputs: None
    '''

    def train_step(self, state, action, reward, next_state, done) -> None:
        state = torch.tensor(state, dtype=torch.float) # converstion to tensor
        next_state = torch.tensor(next_state, dtype=torch.float) # converstion to tensor
        action = torch.tensor(action, dtype=torch.float)  # Keep as float for easier indexing
        reward = torch.tensor(reward, dtype=torch.float) # converstion to tensor

        # check to see if it is batch or not, otherwise converts it so that shape is correct size
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

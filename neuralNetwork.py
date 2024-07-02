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

from actions import *

class CustomEnv:
    def __init__(self):
        self.state = None

    def custom_reset(self):
        self.state = numpy.random.rand(4)  # Initialize state
        return self.state

    def custom_step(self, action):
        if action == 0:
            # buy
            pass
        elif action == 1:
            # upgrade
            pass
        elif action == 2:
            # nothing
            pass
        
        reward = self.calculate_reward()
        done = self.check_done()
        return self.state, reward, done, {}
    def calculate_reward(self):
        # Define how the reward is calculated
        # use money at end of round versus round
        return 1.0
    def close(self):
        pass

    def seed(self, seed=None):
        numpy.random.seed(seed)
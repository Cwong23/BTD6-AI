import torch
import random
import numpy as np
from collections import deque
from neuralNetwork import BTDEnv, Linear_QNet, QTrainer
from actions import restart, resetGame
from helper import plot
import keyboard
import pyautogui
from takeScreenShot import scRound
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft
        self.model = Linear_QNet(input_size=103, hidden_size=64, output_size=6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game: BTDEnv):
        # implement state extraction logic
        state = game.observation
        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
           final_move = [random.randint(0, 2), random.randint(0, 1549), random.randint(0, 919), random.randint(0, 1), random.randint(0, 99), random.randint(0, 2)]
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            action = prediction.detach().tolist()[0]
            
            final_move[0] = int(action[0] * 2)
            final_move[1] = int(action[1] * 1549)
            final_move[2] = int(action[2] * 919)
            final_move[3] = int(action[3] * 1)
            final_move[4] = int(action[4] * 99)
            final_move[5] = int(action[5] * 2)

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = BTDEnv()
    game.reset()
    stop = False

    def on_press_event(e):
        nonlocal stop
        if e.name == 'z':  
            stop = True
            print("Stopping training...")

    keyboard.on_press(on_press_event)
    
    plt.ion()
    fig, ax = plt.subplots()
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("-1100+700")  # Position the window at (100, 100)
    resetGame()
    while True:
        if stop:
            break
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        x = game.step(final_move)
        reward, done, score = x[1], x[2], x[4]
        state_new = agent.get_state(game)
        print("Score: ", score)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            restart()
            pyautogui.sleep(1)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)
            game.rounds = scRound()[0]
            game.rounds = int(game.rounds)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
    plt.ioff()  
    plt.show()


if __name__ == '__main__':
    train()
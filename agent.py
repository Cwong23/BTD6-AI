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
from monkeys import current_monkeys
'''
Notes: Most of this code was from freeCodeCamp's video on Snake RL, but I modified it to fit the correct requirements of my env.
'''

MAX_MEMORY = 100_000
BATCH_SIZE = 25
LR = 0.001

'''
Agent
Purpose: Makes the decisions of what actions to take
Inputs: None
Outputs: Interacts with the env
Logic: Uses various functions and previous games to predict what action to take next
'''

class Agent:
    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft, stores experience tuples
        self.model = Linear_QNet(input_size=103, hidden_size=64, output_size=6) # nn model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # Q trainer

    '''
    get_state(Agent, BTDEnv) -> list
    Purpose: Gets the current state of the game which includes, health, rounds, money, and previous actions
    Inputs: Agent, BTDEnv
    Outputs: list
    '''

    def get_state(self, game: BTDEnv):
        # implement state extraction logic
        state = game.observation 
        return np.array(state, dtype=np.float32)

    '''
    remember(Agent, list, list, float, list, bool) -> None
    Purpose: After an action is finished, it will append to the memory the tuple containing the given info
    Inputs: Agent, list, list, float, list, bool
    Outputs: None
    '''

    def remember(self, state: list, action: list, reward: float, next_state: list, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    '''
    train_long_memory(Agent) -> None
    Purpose: Helps train the nn using a batch of experiences
    Inputs: Agent
    Outputs: None
    Logic: Checks the memory size to see if it should train the long memory with a random selection, otherwise it will use all the experience from memory.
    '''

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE: # checks if there are enough samples
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples, random selection
        else:
            mini_sample = self.memory # uses all of the experience

        states, actions, rewards, next_states, dones = zip(*mini_sample) # creates tuples from the sample
        self.trainer.train_step(states, actions, rewards, next_states, dones) # trains

    '''
    train_short_memory(Agent, list, list, int, list, bool) -> None
    Purpose: Helps train the nn using a single tuple
    Inputs: Agent, list, list, int, list, bool
    Outputs: None
    Logic: Calls the trainer with the info
    '''

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    '''
    get_action(Agent, list) -> list
    Purpose: Gets the next action to be performed
    Inputs: Agent, list
    Outputs: list
    Logic: Decides whether to randomly generate an action, or use a previously generated action. It then creates an action based off that
    '''

    def get_action(self, state: list) -> list:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # checking if it should use a random move
        final_move = [0, 0, 0, 0, 0, 0] # action space

        if random.randint(0, 200) < self.epsilon: # random vs non random
           final_move = [random.randint(0, 2), random.randint(0, 1549), random.randint(0, 919), random.randint(0, 1), random.randint(0, 99), random.randint(0, 2)] # random action
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0) # convert state to a tensor and add dimensions since it expects a batch of inputs
            prediction = self.model(state0) # creates a prediction from the nn based on the current state
            action = prediction.detach().tolist()[0] # detach the prediction tensor and convert it to a list
            
            # generated action
            final_move[0] = int(action[0] * 2) 
            final_move[1] = int(action[1] * 1549)
            final_move[2] = int(action[2] * 919)
            final_move[3] = int(action[3] * 1)
            final_move[4] = int(action[4] * 99)
            final_move[5] = int(action[5] * 2)

        return final_move

'''
train() -> None
Purpose: Trains the agent
Inputs: None
Outputs: None
Logic: Creates the different varaibles to be used. Sets up the agent and env. Creates a function that will be called to stop the training based on a keypress
       Creates a graph and spawns it at a certain location on screen to not interfere with the whole game. Loops infinitely while getting the old state, deciding on
       the next moved based on it, and performing a new action. Then it gets the new state and trains the memory. If the game is completely done, it will train
       the long memory and update the graph.
'''

def train() -> None:
    # variables for plotting 
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # variables for nn
    agent = Agent()
    game = BTDEnv()
    game.reset()
    stop = False

    # used to stop the training
    def on_press_event(e):
        nonlocal stop
        if e.name == 'z':  
            stop = True
            print("Stopping training...")

    keyboard.on_press(on_press_event)
    
    # creates graph at certain location
    plt.ion()
    fig, ax = plt.subplots()
    mng = plt.get_current_fig_manager()
    mng.window.wm_geometry("-1100+700")
    resetGame()

    # infinite loop
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
            current_monkeys.clear()
            pyautogui.sleep(1)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            # create graph
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
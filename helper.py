import matplotlib.pyplot as plt
from IPython import display

'''
Notes: This code was from freeCodeCamp's video on Snake RL. 
'''

'''
plot(scores: list, mean_scores: list) -> None
Purpose: Plot the scores of the games
Inputs: list, list
Outputs: None
Logic: Labels and plots the scores according to the given lists. This shows the score of the last game and the mean score. The score is the
       blue line and the mean score is the orange line.
'''

plt.ion()

def plot(scores: list, mean_scores: list) -> None:
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.1)

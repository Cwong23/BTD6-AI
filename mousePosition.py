import pyautogui
from pyautogui import *

try:
    while True:
        sleep(4)
        x, y = pyautogui.position()
        print("X: ", x, " Y: ", y)
except KeyboardInterrupt:
    print('\n')
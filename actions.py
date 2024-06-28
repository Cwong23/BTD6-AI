import random
import keyboard
from monkeys import Monkey
from pyautogui import *
import pyautogui
from takeScreenShot import scMoney

def buy(x, y, tower: Monkey):
    try:
        money = scMoney()
        keyboard.press_and_release(tower.keybind)
        pyautogui.click(x, y)
        if(scMoney() == money):
            return False
        return True
    except:
        return False

ninja = Monkey("Ninja", 430, 'd')
sleep(3)
print(buy(380, 318, ninja))
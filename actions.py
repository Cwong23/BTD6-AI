import keyboard
from monkeys import Monkey
from pyautogui import *
import pyautogui
from takeScreenShot import scMoney

def buy(x, y, tower: Monkey):
    try:
        print("here")
        money = scMoney()
        keyboard.press_and_release(tower.keybind)
        pyautogui.click(x, y)
        pyautogui.click(x, y)
        pyautogui.sleep(0.25)
        if(scMoney() == money):
            keyboard.press_and_release('esc')
            return False
        pyautogui.click(1623, 1079)
        return True
    except:
        return False

def startRound():
    keyboard.press_and_release(' ')

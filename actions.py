import keyboard
from monkeys import *
from pyautogui import *
import pyautogui
from takeScreenShot import scMoney

def buy(x, y, tower: Monkey):
    try:
        
        money = scMoney()
        keyboard.press_and_release(tower.keybind)
        pyautogui.click(x, y)
        pyautogui.click(x, y)
        pyautogui.sleep(0.25)
        
        if(scMoney() == money):
            keyboard.press_and_release('esc')
            pyautogui.sleep(0.25)
            pyautogui.click(1623, 1079)
            return False
        pyautogui.click(1623, 1079)
        temp = PlacedMonkey(tower, x, y)
        current_monkeys.append(temp)
        
        print("Vro")
        return True
    except:
        print("Something went wrong")
        pyautogui.click(1623, 1079)
        return False

def startRound():
    keyboard.press_and_release(' ')

def upgrade(tower: PlacedMonkey, upgrade: int):
    try:
        money = int(scMoney())
        
        print(tower.upgradeM(upgrade, money))
        if(tower.upgradeM(upgrade, money)):
            
            pyautogui.click(tower.x, tower.y)
            pyautogui.sleep(0.25)
            keyboard.press_and_release(tower.monkey.upgradeKeybinds[upgrade])
            pyautogui.sleep(0.25)
            if(int(scMoney()) == money):
                
                keyboard.press_and_release('esc')
                return False
            pyautogui.click(1623, 1079)
            tower.currentUpgrade[upgrade]+=1
            pyautogui.click(1623, 1079)
            return True
        pyautogui.click(1623, 1079)
        return False
    except:
        pyautogui.click(1623, 1079)
        return False
'''
pyautogui.sleep(2)
d = PlacedMonkey(monkey_dict['ninja'], 349, 218)
d.currentUpgrade = [3, 2, 0]
print(upgrade(d, 1))

pyautogui.sleep(2)
buy(400, 300, monkey_dict["ninja"])

print(current_monkeys) # causing error

pyautogui.sleep(1)
buy(400, 300, monkey_dict["ninja"])
upgrade(current_monkeys[0], 1)
print(current_monkeys[0])
'''


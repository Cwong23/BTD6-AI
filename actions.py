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
            reset()
            return False
        reset()
        temp = PlacedMonkey(tower, x, y)
        current_monkeys.append(temp)
        
        
        return True
    except:
        print("Buy failed")
        reset()
        return False

def startRound():
    keyboard.press_and_release(' ')

def restart():
    pyautogui.click(850, 775)
    pyautogui.sleep(0.25)
    pyautogui.click(1115, 718)

def reset():
    pyautogui.sleep(0.15)
    keyboard.press_and_release('esc')
    pyautogui.sleep(0.15)
    pyautogui.click(1623, 1079)

# work on upgrade math

def upgrade(tower: PlacedMonkey, upgrade: int):
    try:
        pyautogui.sleep(0.5)
        money = int(scMoney())
        print("check 1")
        if(tower.upgradeM(upgrade, money)):
            print("check 2")
            pyautogui.click(tower.x, tower.y)
            pyautogui.sleep(0.25)
            keyboard.press_and_release(tower.monkey.upgradeKeybinds[upgrade])
            reset()
            for i in range(5):
                money2 = scMoney()
                if(money2 != ""):
                    if(int(money2) == money):
                        reset()
                        return False
                    reset()
                    tower.currentUpgrade[upgrade]+=1
                    return True 
                print("failed to read money")
        reset()
        return False
    except:
        reset()
        print("Upgrade failed")
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


pyautogui.sleep(1)
buy(400, 300, monkey_dict["ninja"])
print(upgrade(current_monkeys[0], 0))
'''



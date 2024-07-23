import keyboard
from monkeys import *
from pyautogui import *
import pyautogui
from takeScreenShot import scMoney

'''
Purpose: Enable the agent to interact with the game BTD 6 using hotkeys and specific locations
'''

'''
buy(int, int, Monkey) -> bool
Purpose: Buy a tower at a specific location.
Inputs: int, int, Monkey
Outputs: bool
Logic: Given the x and y coordinates and the tower to be bought, it takes a note of the current money, then buys the tower at that location.
       After, it checks the money again to see if it went down, if it did then the buy was successful, otherwise it returns false. If the buy 
       was successful, it will add the tower to the list holding all of the current towers on the map.
'''

def buy(x: int, y: int, tower: Monkey) -> bool:
    try:
        money = scMoney() # takes note of money
        keyboard.press_and_release(tower.keybind) # keybind press to select the monkey
        pyautogui.click(x, y) # clicks twice in case of error to place monkey
        pyautogui.click(x, y)
        pyautogui.sleep(0.25) # sleep to allow for money to update
        
        if(scMoney() == money): # checks money
            resetGame() # buy was unsuccessful
            return False
        resetGame() # reset the position of mouse and any open screens
        temp = PlacedMonkey(tower, x, y) # add to the list
        current_monkeys.append(temp)
        
        
        return True
    except:
        print("Buy failed")
        resetGame()
        return False

'''
startRound() -> none
Purpose: Start the round in the game.
Inputs: none
Outputs: none
Logic: press and release the space key to start the round
'''

def startRound():
    keyboard.press_and_release(' ')
    pyautogui.sleep(1)

'''
restart() -> none
Purpose: Restart the game after defeat
Inputs: none
Outputs: none
Logic: clicks on specific positions and waits to start the game
'''

def restart():
    pyautogui.click(850, 775)
    pyautogui.sleep(0.25)
    pyautogui.click(1115, 718)
    pyautogui.sleep(0.25)

'''
resetGame() -> none
Purpose: Reset the mouse position and other open menus
Inputs: none
Outputs: none
Logic: By doing this specific sequence of key presses and clicks, it will completely reset the screen to a valid action position
'''

def resetGame():
    pyautogui.sleep(0.15)
    keyboard.press_and_release('esc')
    pyautogui.sleep(0.15)
    pyautogui.click(1623, 1079)

'''
upgrade(PlacedMonkey, int) -> bool
Purpose: Upgrade a specific tower given an upgrade path
Inputs: PlacedMonkey, int
Outputs: bool
Logic: It will take note of current money. Then it will select the given placed tower and attempt to upgrade it. It will also attempt to read money
       multiple times since a new window pops up that distrubs the location of the money on screen. It will then compare the money with the new money
       to see if the upgrade was successful.
'''

def upgrade(tower: PlacedMonkey, upgrade: int) -> bool:
    try:
        pyautogui.sleep(0.5)
        money = int(scMoney()) # reads current money
        
        if(tower.upgradeM(upgrade, money)): # calls the tower to check if the upgrade is valid
            
            pyautogui.click(tower.x, tower.y) # goes for the upgrade
            pyautogui.sleep(0.25)
            keyboard.press_and_release(tower.monkey.upgradeKeybinds[upgrade])
            resetGame() # reset mouse position
            for i in range(5): # attempts to read money again
                money2 = scMoney()
                if(money2 != ""):
                    if(int(money2) == money):
                        resetGame()
                        return False
                    resetGame() # if it is successful, then the tower will have it's upgrade paths updated
                    tower.currentUpgrade[upgrade]+=1
                    return True 
                print("failed to read money")
        resetGame()
        return False
    except:
        resetGame()
        print("Upgrade failed")
        return False
    




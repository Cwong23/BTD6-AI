'''
Purpose: Data Stuctures and classes to hold information regarding the towers
'''

'''
Monkey
Purpose: Hold the name, price, keybind, and upgrade paths of a tower
Inputs: str, int, str, list
Output: None
'''

class Monkey:
    def __init__(self, name: str, price: int, keybind: str, upgrades: list) -> None:
        self.name = name
        self.price = price
        self.keybind = keybind
        self.upgrades = upgrades # 1(1->5) 2(1->5) 3(1->5)
        self.upgradeKeybinds = [',', '.', '/']
    
'''
PlacedMonkey
Purpose: Hold information regarding a monkey, it's current upgrades, and it's position in the game
Inputs: Monkey, x, y
Output: None
'''

class PlacedMonkey:
    def __init__(self, monkey: Monkey, x: int, y: int) -> None:
        self.currentUpgrade = [0, 0, 0]
        self.monkey = monkey
        self.x = x
        self.y = y

    '''
    upgradeM(self, upgradePath: int, money: int) -> bool
    Purpose: Check if an upgrade is possible
    Inputs: PlacedMonkey, int, int
    Output: bool
    Logic: Checks if the the upgrade paths don't already have 2 open because then the third path is closed. Then checks if
           the other path is less than 3. Then checks for money.
    '''

    def upgradeM(self, upgradePath: int, money: int) -> bool:
        pathCheck = 0
        
        # check if the upgrade is possible
        for i,x in enumerate(self.currentUpgrade): # checks for the validity of path based on other paths
            if x != 0:
                pathCheck+=1
            
            if x >= 3 and i != upgradePath: # checks for a path that has more than 3 but not the intended upgrade path
                return False
       
        if (self.currentUpgrade[upgradePath] == 0) and (pathCheck == 2): # checks if the other paths were opened already
            return False
        
        # money check
        if(self.monkey.upgrades[(upgradePath*5)+self.currentUpgrade[upgradePath]] > money): 
            return False
        
        print("Cost: " + str(self.monkey.upgrades[(upgradePath*5)+self.currentUpgrade[upgradePath]]))
        return True
    
    def __str__(self) -> str:
        rStr = self.monkey.name + " X: " + str(self.x) + " Y: " + str(self.y) + " Upgrades: " + str(self.currentUpgrade[0]) + str(self.currentUpgrade[1]) + str(self.currentUpgrade[2])
        return rStr


'''
monkey_dict{}
Purpose: Holds the generic towers in a dict. This allows for easy configuration based on mode or what towers should be restricted
'''
# Currently configured for Easy Mode
monkey_dict = {
    0: Monkey("ninja", 340, 'd', [295, 295, 720, 2335, 29750, 295, 425, 765, 4420, 18700, 210, 340, 2335, 3825, 34000]),
    1: Monkey("bomb", 445, 'e', [295, 550, 1020, 3060, 46750, 210, 340, 930, 2720, 21250, 170, 255, 680, 2380, 29750])
}
'''
current_monkeys[]]
Purpose: Holds all the placed towers
'''
current_monkeys = []
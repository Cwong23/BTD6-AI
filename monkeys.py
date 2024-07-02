class Monkey:
    def __init__(self, name: str, price: int, keybind, upgrades) -> None:
        self.name = name
        self.price = price
        self.keybind = keybind
        self.upgrades = upgrades # 1(1->5) 2(1->5) 3(1->5)
        self.upgradeKeybinds = [',', '.', '/']
    

class PlacedMonkey:
    def __init__(self, monkey: Monkey, x: int, y: int) -> None:
        self.currentUpgrade = [0, 0, 0]
        self.monkey = monkey
        self.x = x
        self.y = y

    def upgradeM(self, upgradePath: int, money: int):
        pathCheck = 0
        print("there")
        # check if the upgrade is possible
        for i,x in enumerate(self.currentUpgrade):
            if x != 0:
                pathCheck+=1
            
            if x >= 3 and i != upgradePath:
                return False
       
        if (self.currentUpgrade[upgradePath] == 0) and (pathCheck == 2):
            return False
        print("here")
        # money check
        print((upgradePath*5)+self.currentUpgrade[upgradePath])
        if(self.monkey.upgrades[(upgradePath*5)+self.currentUpgrade[upgradePath]] > money): 
            
            return False
        print("Cost: " + str(self.monkey.upgrades[(upgradePath*5)+self.currentUpgrade[upgradePath]]))
        return True
    
    def __str__(self) -> str:
        rStr = self.monkey.name + " X: " + str(self.x) + " Y: " + str(self.y) + " Upgrades: " + str(self.currentUpgrade[0]) + str(self.currentUpgrade[1]) + str(self.currentUpgrade[2])
        return rStr

monkey_dict = {
    'ninja': Monkey("ninja", 340, 'd', [295, 295, 720, 2335, 29750, 295, 425, 765, 4420, 18700, 210, 340, 2335, 3825, 34000]),
    'bomb': Monkey("bomb", 445, 'e', [295, 550, 1020, 3060, 46750, 210, 340, 930, 2720, 21250, 170, 255, 680, 2380, 29750])
}

current_monkeys = []
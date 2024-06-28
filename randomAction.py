import random
from actions import *
from monkeys import monkey_dict

def randomAction(money):
    x = random.randint(1,2)
    y = random.randint(1,1)
    if(x==1):
        print("Placing Ninja")
        monkeyToPlace = monkey_dict["ninja"]
    else:
        print("Placing Bomb")
        monkeyToPlace = monkey_dict["bomb"]

    if(y == 1) and int(money) >= monkeyToPlace.price:
        
        for i in range(5):
            
            p = random.randint(50, 1700)
            o = random.randint(50, 900)
            print("X: ", p, "Y: ", o)
            if(buy(p, o, monkeyToPlace)):
                break
            print("Failed to place")
            
            pyautogui.sleep(1)
    else:
        print("Not enough money")
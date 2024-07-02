import random
from actions import *
from monkeys import *

def randomAction(money):
    x = random.randint(1,2)
    y = random.randint(1,2)
    if(y==1):
        if(x==1):
            print("Placing Ninja")
            monkeyToPlace = monkey_dict["ninja"]
        else:
            print("Placing Bomb")
            monkeyToPlace = monkey_dict["bomb"]

        if(int(money) >= monkeyToPlace.price):
            
            for i in range(5):
                
                p = random.randint(50, 1600)
                o = random.randint(80, 1000)
                print("X: ", p, "Y: ", o)
                if(buy(p, o, monkeyToPlace)):
                    print("Bought")
                    break
                print("Failed to place")
                
                pyautogui.sleep(1)
        else:
            print("failed to buy")
   
    elif(y==2):
        p = random.randint(0,2)
        print("Upgrade Path ", p)
        
        if(len(current_monkeys) != 0):
            z = random.randint(0, len(current_monkeys)-1)
            print("Monkey Chosen: ")
            print(current_monkeys[z])
            upgrade(current_monkeys[z], p)
            print("Monkey Now: ")
            print(current_monkeys[z])
        else:
            print("failed to upgrade")
    
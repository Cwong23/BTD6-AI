import tkinter as tk
from testing.randomAction import *
from takeScreenShot import *

class GUIApp:
    def __init__(self, root):

        self.money = scMoney()
        self.rounds = scRound()
        self.health = scLives()

        self.root = root
        self.root.title("Current Info")

        # create a label to display information
        self.info_label = tk.Label(self.root, text="Current Info", padx=10, pady=10)
        self.info_label.pack()

        self.update_interval = 3000  # update interval in milliseconds

        end_button = tk.Button(root, text="Stop", command=self.stop_app)
        end_button.pack()

        self.is_running = True 
        print("Starting...")
        self.update_info() # function call to update info

    def stop_app(self): # ends the program
        self.root.destroy()

    def update_info(self):
        if not self.is_running: # checks if it is supposed to run
            return
        
        if not scCurrent(): # checks so that it only updates when the round is over
            # update the label with all pieces of information
            # all of these call functions in takeScreenShot.py

            for i in range(5):
                self.money = scMoney()
                if(self.money != ""):
                    randomAction(self.money)
                    pyautogui.sleep(4)
                    break
                print("failed to read money")
            self.money = scMoney()
            self.rounds = scRound()
            self.health = scLives()
            pyautogui.sleep(2)
            startRound()

        info_text = f"Money: {self.money}\n" \
                    f"Rounds: {self.rounds}\n" \
                    f"Health: {self.health}\n" \
                    f"Live: {scCurrent()}\n" \
                    f"Action: {scLives()}"
        
        self.info_label.config(text=info_text)



        # schedule the next update
        self.root.after(self.update_interval, self.update_info)





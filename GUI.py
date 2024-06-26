import tkinter as tk
from takeScreenShot import *

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Current Info")

        # create a label to display information
        self.info_label = tk.Label(self.root, text="Current Info", padx=10, pady=10)
        self.info_label.pack()

        self.update_interval = 2000  # update interval in milliseconds

        self.is_running = True 
        self.update_info() # function call to update info

    def update_info(self):
        if not self.is_running: # checks if it is supposed to run
            return
        
        # update the label with all pieces of information
        # all of these call functions in takeScreenShot.py
        info_text = f"Money: {scMoney()}\n" \
                    f"Rounds: {scRound()}\n" \
                    f"Health: {scLives()}\n" \
                    f"Live: {scCurrent()}"
        
        self.info_label.config(text=info_text)

        # schedule the next update
        self.root.after(self.update_interval, self.update_info)





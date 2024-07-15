import pyautogui
import time
from imageProcessing import processImage
import keyboard

'''
Purpose: Take screenshots of certain information elements and then return the data gathered from the images
'''

'''
scMoney(), scLives(), scRound()
Purpose: Get info out of the screenshots taken.
Inputs: None
Outputs: A string
Logic: Take a screenshot of a certain area. Process the image and return.
'''
def scMoney():
    pyautogui.screenshot('screenshots\screenshotMoney.jpg', region=(365,24, 170, 40)) # left, top, width, and height
    x = processImage('screenshots\screenshotMoney.jpg', "1")
    if x == '':
        return '-1'
    return x

def scLives():
    pyautogui.screenshot('screenshots\screenshotLives.jpg', region=(139, 20, 100, 40))
    x = processImage('screenshots\screenshotLives.jpg', "2")
    if x == '':
        return '-1'
    return x

def scRound():
    pyautogui.screenshot('screenshots\screenshotRound.jpg', region=(1420, 32, 140, 50))
    x = processImage('screenshots\screenshotRound.jpg', "3").split('/')
    if x[0] == '':
        return ['-1']
    return x



'''
scCurrent()
Purpose: See if the round is running or not.
Inputs: None
Outputs: Boolean
Logic: Take a screenshot of a certain area. Compare that image to another image to see if the round if live or not. If the play button is on screen, the round is not live.
'''
def scCurrent(): # returns false if the game is running
    pyautogui.screenshot('screenshots\screenshotCurrent.jpg', region=(1800, 970, 75, 50))
    return(open("screenshots\screenshotCurrent.jpg","rb").read() != open("screenshots\compareCurrent.jpg","rb").read())

def scDef(): 
    pyautogui.screenshot('screenshots\screenshotDefeat.jpg', region=(800, 775, 75, 75))
    return(open("screenshots\screenshotDefeat.jpg","rb").read() != open("screenshots\compareDefeat.jpg","rb").read())

def scWin():
    pyautogui.screenshot('screenshots\screenshotWin.jpg', region=(700, 165, 300, 50))
    return(open("screenshots\screenshotWin.jpg","rb").read() != open("screenshots\compareWin.jpg","rb").read())


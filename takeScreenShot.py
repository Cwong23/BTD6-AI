import pyautogui
import time
from imageProcessing import processImage
import keyboard

#money coords 
def scMoney():
    pyautogui.screenshot('screenshots\screenshotMoney.jpg', region=(370,20, 150, 40))
    return processImage('screenshots\screenshotMoney.jpg')

def scLives():
    pyautogui.screenshot('screenshots\screenshotLives.jpg', region=(139, 20, 100, 40))
    return processImage('screenshots\screenshotLives.jpg')

def scRound():
    pyautogui.screenshot('screenshots\screenshotRound.jpg', region=(1450, 32, 110, 50))
    return processImage('screenshots\screenshotRound.jpg').split('/')

def scWin():
    pyautogui.screenshot('screenshots\screenshotWin.jpg', region=(900, 888, 121, 55))
    return processImage('screenshots\screenshotWin.jpg')

def scCurrent():
    pyautogui.screenshot('screenshots\screenshotCurrent.jpg', region=(1800, 970, 75, 50))
    return(open("screenshots\screenshotCurrent.jpg","rb").read() == open("screenshots\compareCurrent.jpg","rb").read())
print(scCurrent())
print(scMoney())
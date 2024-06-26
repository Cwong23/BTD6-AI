import cv2
import pytesseract
import numpy as np
from PIL import Image
from pytesseract import Output

'''

Purpose: Get data out of images
Notes: A majority of this code was taken from NeuralNine, I have attached the video that I used in the README.md.

'''



'''
noise_removal(image)
Puprose: Remove noise from an image
Inputs: An image
Output: An image with noise removed
Logic: Using cv2 functions, I am able to remove noise from the image
'''
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8) # used in the other functions
    image = cv2.dilate(image, kernel, iterations=1) # adds dilation pixels to help close small holes/broken parts in the image
    image = cv2.erode(image, kernel, iterations=1) # removes boundary pixels on objects
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) # closes small holes and gaps
    image = cv2.medianBlur(image, 3) # removes salt and pepper noise
    return image

'''
processImage(process_This, numb)
Puprose: Gather data from an image
Inputs: An image, a string number corresponding to the image data
Output: A string containing info from the image
Logic:  Open the image and remove as much noise as possible by gray scaling and using the 'noise_removal(image)' function. 
        Using Tesseract, I then isolate digits and convert the image to string text. Finally, I concat a string and return it.
'''

def processImage(process_This, numb):
    im = Image.open(process_This) # opens image
    im.save(process_This, dpi=(300, 300)) # saves image
    image = cv2.imread(process_This) # reads image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray scales

    # define lower and upper bounds for white and light gray
    lower_bound = np.array([200], dtype="uint8")
    upper_bound = np.array([255], dtype="uint8")

    mask = cv2.inRange(gray, lower_bound, upper_bound) # create color mask using gray scale bounds

    result = cv2.bitwise_and(gray, gray, mask=mask) # apply the mask keeping only numbers in bounds

    no_noise = noise_removal(result) # noise removal

    processed_image_path = "screenshots/strippedImage" + numb +".jpg" # save stripped image for debugging purposes
    cv2.imwrite(processed_image_path, no_noise) # write to the saved image

    myconfig = r"--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/" # Tesseract configuration for reading numbers and / for the rounds

    data = pytesseract.image_to_string(no_noise, lang='eng', config=myconfig) # extract text using Tesseract

    # Filter text by confidence
    returnThis = ""
    data_dict = pytesseract.image_to_data(no_noise, config=myconfig, output_type=Output.DICT)
   
    # loops through the gathered data, and checks for the confidence level
    for i in range(len(data_dict['text'])):
        if float(data_dict['conf'][i]) > 5:
            if(numb != "3"): # if not rounds, then get rid of everything that is not a digit
                if data_dict['text'][i].isdigit():
                    returnThis += data_dict['text'][i]
            else: # if it is rounds, then keep the '/' and splice it out later
                returnThis += data_dict['text'][i]
    return returnThis



import cv2
import pytesseract
import numpy as np
from PIL import Image
from pytesseract import Output


# Noise removal
def noise_removal(image):
    kernel =np.ones((1,1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image,3)
    return(image)

def processImage(process_This):
    # Color Inversion, Grayscale, Gaussian blur, Otsu's threshold
    im = Image.open(process_This)
    im.save(process_This, dpi=(300,300))
    image = cv2.imread(process_This)
    whiteLower = np.array([247, 247, 247], dtype = "uint8")
    whiteUpper = np.array([255, 255, 255], dtype = "uint8")
    white = cv2.inRange(image, whiteLower, whiteUpper)
    inversion = cv2.bitwise_not(white)
    
    blur = cv2.GaussianBlur(inversion, (3,3), 0)
    thresh = cv2.threshold(blur, 210, 230, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    
   
    no_noise = noise_removal(thresh)
    cv2.imwrite("screenshots\strippedImage.jpg", no_noise)
    # Data Collection

    myconfig = r"--psm 11 --oem 3"
    data = pytesseract.image_to_string(no_noise, lang='eng', config=myconfig)
    #print("data: ",data)
    
    returnThis = ""
    """
    #get rid of other things besides numbers
    for x in data:
        try:
            int(x)
            returnThis+=x
        except:
            j = 1
    """
    data = pytesseract.image_to_data("screenshots\strippedImage.jpg", config='--psm 11', output_type=Output.DICT)
    for i in range(len(data['text'])):
        if float(data['conf'][i]) > 20:
            returnThis+=data['text'][i]
    return returnThis



    
   

    




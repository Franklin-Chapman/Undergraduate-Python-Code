#program imports in a photo and takes takes the brightness measurements 
#of an image one pixel at a time
#will proform the brightness measurement accross a few chosen lines
#will then take those measurments and find the contrast by taking a ratio
#will then plot the brightness measurments

import numpy as np
from PIL import Image
from math import sqrt
imag = Image.open("red.JPG")
#Convert the image te RGB if it is a .gif for example
imag = imag.convert ('RGB')
#coordinates of the pixel
X = np.array([1])
Y = np.array([1])
#X,Y = 0,0
#X,Y = list((100, 45), (80, 45))
#Get RGB

for i in range(X,Y):
    pixelRGB = imag.getpixel((X,Y))

#pixelRGB = imag.getpixel((X,Y))
R,G,B = pixelRGB 
#combine R,G,B brightness to get total brightness of the pixel in question
brightness = sum([R,G,B])/3 ##0 is dark (black) and 255 is bright (white)

#brightness is simply the sum of the brightness in the different colors
print(brightness)
LuminanceA = (0.2126*R) + (0.7152*G) + (0.0722*B)
#luminance is the intensity of a specific measured spot
#we will go with luminance
print(LuminanceA)
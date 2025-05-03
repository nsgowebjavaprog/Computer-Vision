import cv2 as cv
import numpy as np


blank = np.zeros((500, 500, 3), dtype='uint8')  # 3 channels for color
cv.imshow('Blank', blank)


# Draw a green rectangle on the blank image
blank[200:300, 300:400] = (0, 255, 0)  # BGR format
cv.imshow('Green', blank)



cv.rectangle(blank, (0, 0), (250, 500), (0, 255,0), thickness=cv.FILLED)  
cv.imshow('Rectangle', blank)

# Circle

cv.circle(blank, (250, 250), 40, (255, 0, 0), thickness=cv.FILLED)
cv.imshow('Circle', blank)


# Line

cv.line(blank, (0, 0), (250, 500), (0, 0, 255), thickness=3)
cv.imshow('Line', blank)

# Text

cv.putText(blank, 'NS LONI', (225,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), thickness=2)
cv.imshow('Text', blank)


cv.waitKey(0)
cv.destroyAllWindows()

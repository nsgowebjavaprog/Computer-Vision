import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR to Gray
cv.imshow('Gray', gray)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # BGR to HSV
cv.imshow('HSV', hsv)

lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)  # BGR to Lab
cv.imshow('Lab', lab)


rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # BGR to RGB
cv.imshow('RGB', rgb)

lab_rgb = cv.cvtColor(lab, cv.COLOR_Lab2RGB)  # Lab to RGB
cv.imshow('Lab to RGB', lab_rgb)

# plt.imshow(img)  
# plt.show()
cv.waitKey(0)
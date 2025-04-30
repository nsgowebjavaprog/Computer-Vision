import cv2 as cv
img = cv.imread('Resources/Photos/cat.jpg')   # forward slashes

cv.imshow('cat', img)

cv.waitKey(0)
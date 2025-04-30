import cv2 as cv
import numpy as np

# Read and display the original image
img = cv.imread('Resources/Photos/park.jpg')
cv.imshow('Original', img)

# Split image into its color channels
b, g, r = cv.split(img)

# Create a blank image with same height and width (for merging channels later)
blank = np.zeros(img.shape[:2], dtype='uint8')

# Display individual color channels
cv.imshow('Blue', b)
cv.imshow('Green', g)
cv.imshow('Red', r)

# Merge individual channels with blank to visualize each color
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue Channel', blue)
cv.imshow('Green Channel', green)
cv.imshow('Red Channel', red)

# Print shape of original and channel images
print("Original shape:", img.shape)
print("Blue shape:", b.shape)
print("Green shape:", g.shape)
print("Red shape:", r.shape)

# Merge back the channels to form original image
merged = cv.merge([b, g, r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)
cv.destroyAllWindows()
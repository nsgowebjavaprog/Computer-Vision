import cv2 as cv

# # Reading an image using OpenCV

import cv2 as cv
img = cv.imread('Resources/Photos/cat_large.jpg')   # forward slashes

cv.imshow('cat', img)

cv.waitKey(0)

# #----------------------------------------------------------------------

# Reading an Video using OpenCV

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()  # isTrue = True if the video is opened successfully
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()  
cv.destroyAllWindows()


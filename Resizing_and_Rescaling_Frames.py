#---------- Resizing and Rescaling Frames ------------


import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)  
    height = int(frame.shape[0] * scale)
    
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)



capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()  # isTrue = True if the video is opened successfully
    
    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Resized Video', frame_resized)  
    
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()  
cv.destroyAllWindows()
import cv2
import numpy as np

def check_quality(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    full_mask = mask1 + mask2
    red_pixels = cv2.countNonZero(full_mask)
    total_pixels = image.shape[0] * image.shape[1]

    red_ratio = red_pixels / total_pixels
    if red_ratio > 0.5:
        return 'High'
    elif red_ratio > 0.3:
        return 'Medium'
    else:
        return 'Low'

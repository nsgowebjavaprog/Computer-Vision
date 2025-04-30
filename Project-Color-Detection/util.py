import cv2
import numpy as np

def get_limits(color):
    """
    Given a BGR color, return the lower and upper HSV limits.
    """
    c = np.uint8([[color]])  # BGR to HSV
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]

    lower = np.array([max(hue - 10, 0), 100, 100])
    upper = np.array([min(hue + 10, 179), 255, 255])

    return lower, upper

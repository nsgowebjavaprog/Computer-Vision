import cv2
from PIL import Image
from util import get_limits

# Define at least 10 colors in BGR format with their names
colors_to_detect = {
    'Yellow':  [0, 255, 255],
    'Red':     [0, 0, 255],
    'Blue':    [255, 0, 0],
    'Green':   [0, 255, 0],
    'Orange':  [0, 165, 255],
    'Cyan':    [255, 255, 0],
    'Magenta': [255, 0, 255],
    'Pink':    [203, 192, 255],
    'Purple':  [128, 0, 128],
    'White':   [255, 255, 255]
}

cap = cv2.VideoCapture(0)  # Try changing to 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("❌ Error: Cannot access the camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame.")
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through all colors
    for color_name, bgr in colors_to_detect.items():
        lowerLimit, upperLimit = get_limits(color=bgr)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, f"{color_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    cv2.imshow('10+ Color Detection - Press Q to Exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()

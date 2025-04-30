import cv2
import joblib
from utils.size_detector import get_pomegranate_size
from utils.quality_checker import check_quality
from tabulate import tabulate
import time

def capture_from_camera(output_path='captured.jpg'):
    print("📸 Opening camera. Press 's' to capture and 'q' to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow('Pomegranate Capture', frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite(output_path, frame)
            print(f"✅ Image saved as {output_path}")
            break
        elif key == ord('q'):
            print("❌ Quit without capturing.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path

def predict_price(image_path, weight):
    size = get_pomegranate_size(image_path)
    quality = check_quality(image_path)

    model = joblib.load('pomegranate_price_model.pkl')
    le = joblib.load('label_encoder.pkl')
    quality_encoded = le.transform([quality])[0]

    features = [[size, weight, quality_encoded]]
    predicted_price = model.predict(features)[0]

    return {
        "Size (pixels)": size,
        "Quality": quality,
        "Predicted Price (INR)": round(float(predicted_price), 2)
    }

if __name__ == "__main__":
    image_path = capture_from_camera()  # ← Webcam capture
    weight = 400  # You can make this dynamic or estimated
    result = predict_price(image_path, weight)

    table = [[k, v] for k, v in result.items()]
    print(tabulate(table, headers=["Attribute", "Value"], tablefmt="fancy_grid"))

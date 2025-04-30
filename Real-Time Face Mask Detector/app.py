import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, Response
import mlflow
import mlflow.keras

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Face Mask Detection model (You can train it separately and save as 'mask_detector.model')
MODEL_PATH = 'mask_detector.h5'
model = load_model(MODEL_PATH)

# Define face mask detection function
def detect_mask(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through each face detected
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))  # Resize for CNN model input
        face = np.expand_dims(face, axis=0)
        face = face / 255.0  # Normalize the image
        
        # Predict using the model
        prediction = model.predict(face)
        label = 'Mask' if prediction[0][0] > 0.5 else 'No Mask'
        
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)  # Green for mask, Red for no mask
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Video streaming generator function
def generate_frames():
    # Start webcam feed
    camera = cv2.VideoCapture(0)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Detect mask in the frame
        frame = detect_mask(frame)
        
        # Encode frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to track the model with MLflow
@app.route('/track_model')
def track_model():
    with mlflow.start_run():
        mlflow.keras.log_model(model, "face_mask_detector_model")
    return "Model is tracked successfully!"

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)

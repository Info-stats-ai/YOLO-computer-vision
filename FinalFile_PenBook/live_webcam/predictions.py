from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Run real-time prediction from webcam (0 = default webcam)
model.predict(source=0, show=True, conf=0.15)

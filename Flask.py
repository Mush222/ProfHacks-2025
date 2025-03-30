from flask import Flask, render_template, Response
import cv2
import numpy as np
import winsound
import threading
import time

app = Flask(__name__)

# Initialize video capture (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Load multiple face detection classifiers for different angles
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Video stream function
def generate_frames():
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw boxes around detected faces
    for (x, y, w, h) in faces_frontal:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in faces_profile:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert image to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/video_feed')
def video_feed():
  return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
  app.run(debug=True)

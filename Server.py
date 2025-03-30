import cv2
import time
import json
from flask import Flask, render_template, jsonify, request, Response
from threading import Thread
import winsound

app = Flask(__name__)

# Variable to control the camera process
camera_running = False
cap = None  # Video capture will be initialized dynamically

# Load multiple face detection classifiers for different angles
face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

interval_data = {
  'intervalAverage': 0.0,
  'lastInterval': 0.0,  # Store the last interval between beeps
  'lastBeepTime': 0
}

# Function to handle face detection and beep when the face is too close
def face_detection():
  global camera_running, cap, interval_data
  last_beep_time = 0
  beep_interval = 1.0  # Minimum time between beeps in seconds
  start_time = time.time()
  interval_sum = 0
  times_beeped = 0

  print("Face detection started")  # Debug log

  while camera_running:
    ret, frame = cap.read()
    if not ret:
      print("Error: Can't receive frame")
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal faces
    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Detect profile faces
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Detect faces in flipped image (profile faces from the opposite direction)
    faces_profile_flipped = face_cascade_profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Check if a face is detected
    face_detected = len(faces_frontal) > 0 or len(faces_profile) > 0 or len(faces_profile_flipped) > 0
    distance = 0

    if len(faces_frontal) > 0:
      for (x, y, w, h) in faces_frontal:
        distance = 100 - (w * 0.3)  # Estimate the distance
        if distance <= 55:
          current_time = time.time()
          if current_time - last_beep_time >= beep_interval:
            winsound.Beep(650, 500)  # Beep when distance is less than or equal to 55
            
            # Calculate interval since last beep
            interval = current_time - start_time
            interval_sum += interval
            times_beeped += 1
            interval_data['intervalAverage'] = interval_sum / times_beeped
            interval_data['lastInterval'] = interval  # Store the last interval
            interval_data['lastBeepTime'] = current_time
            
            # Update timing variables
            start_time = current_time
            last_beep_time = current_time
            
            # Debug log
            print(f"Beep! Interval: {interval:.2f}s, Average: {interval_data['intervalAverage']:.2f}s")

    elif len(faces_profile) > 0:
      for (x, y, w, h) in faces_profile:
        distance = 100 - (w * 0.3)
        if distance <= 55:
          current_time = time.time()
          if current_time - last_beep_time >= beep_interval:
            winsound.Beep(1000, 500)
            
            # Calculate interval since last beep
            interval = current_time - start_time
            interval_sum += interval
            times_beeped += 1
            interval_data['intervalAverage'] = interval_sum / times_beeped
            interval_data['lastInterval'] = interval  # Store the last interval
            interval_data['lastBeepTime'] = current_time
            
            # Update timing variables
            start_time = current_time
            last_beep_time = current_time

    elif len(faces_profile_flipped) > 0:
      for (x, y, w, h) in faces_profile_flipped:
        x = frame.shape[1] - x - w  # Adjust x-coordinate for flipped face detection
        distance = 100 - (w * 0.3)
        if distance <= 55:
          current_time = time.time()
          if current_time - last_beep_time >= beep_interval:
            winsound.Beep(1000, 500)
            
            # Calculate interval since last beep
            interval = current_time - start_time
            interval_sum += interval
            times_beeped += 1
            interval_data['intervalAverage'] = interval_sum / times_beeped
            interval_data['lastInterval'] = interval  # Store the last interval
            interval_data['lastBeepTime'] = current_time
            
            # Update timing variables
            start_time = current_time
            last_beep_time = current_time

    time.sleep(0.1)

# Flask routes
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/get_interval')
def get_interval():
  return jsonify(interval_data)

@app.route('/start_detection')
def start_detection():
  global camera_running, cap
  if not camera_running:
    camera_running = True
    cap = cv2.VideoCapture(0)  # Initialize camera capture
    if not cap.isOpened():
      return jsonify({'status': 'Error: Cannot access camera'}), 500
    thread = Thread(target=face_detection)
    thread.start()
  return jsonify({'status': 'Started face detection'})

@app.route('/stop_detection')
def stop_detection():
  global camera_running, cap
  camera_running = False
  if cap is not None:
    cap.release()
    cv2.destroyAllWindows()
  return jsonify({'status': 'Stopped face detection'})

@app.route('/listen_updates')
def listen_updates():
  def event_stream():
    last_sent = 0
    while True:
      try:
        # Only send updates if there's new data
        current_data = interval_data.copy()  # Create a copy to avoid race conditions
        if current_data['lastBeepTime'] > last_sent:
          last_sent = current_data['lastBeepTime']
          data_str = json.dumps(current_data)
          yield f"data: {data_str}\n\n"
        time.sleep(0.5)
      except Exception as e:
        print(f"Error in event stream: {e}")
        continue

  return Response(event_stream(), mimetype="text/event-stream", headers={
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no'
  })

# Shut down the Flask server properly
@app.route('/shutdown')
def shutdown():
  global camera_running, cap
  camera_running = False
  if cap is not None:
    cap.release()
    cv2.destroyAllWindows()
  shutdown_func = request.environ.get('werkzeug.server.shutdown')
  if shutdown_func:
    shutdown_func()
  return 'Server shutting down...'

if __name__ == '__main__':
  app.run(debug=True)

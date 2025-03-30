from PIL import Image
import winsound
import time

app = Flask(__name__)

# The main function you provided, modified to be callable
def run_face_detection():
  cap = cv2.VideoCapture(0)
  face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

  if not cap.isOpened():
    print("Error: Could not open camera")
    return "Error: Camera initialization failed"

  print("Multi-Angle Face Detection System Initialized")

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Error: Can't receive frame")
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_frontal = face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    faces_profile = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    faces_profile_flipped = face_cascade_profile.detectMultiScale(cv2.flip(gray, 1), scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    face_detected = len(faces_frontal) > 0 or len(faces_profile) > 0 or len(faces_profile_flipped) > 0

    if len(faces_frontal) > 0:
      for (x, y, w, h) in faces_frontal:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        distance = 100 - (w * 0.3)
        distance = max(0, min(100, distance))
        cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if face_detected and (distance <= 20):
          winsound.Beep(1000, 500)

    if len(faces_profile) > 0:
      for (x, y, w, h) in faces_profile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        distance = 100 - (w * 0.3)
        distance = max(0, min(100, distance))
        cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if face_detected and (distance <= 20):
          winsound.Beep(1000, 500)

    if len(faces_profile_flipped) > 0:
      for (x, y, w, h) in faces_profile_flipped:
        x = frame.shape[1] - x - w
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        distance = 100 - (w * 0.3)
        distance = max(0, min(100, distance))
        cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if face_detected and (distance <= 20):
          winsound.Beep(1000, 500)

    cv2.imshow('Multi-Angle Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

@app.route('/')
def index():
  # Call the face detection function
  run_face_detection()
  return render_template('index.html')  # Ensure index.html exists in your templates folder

if __name__ == "__main__":
  app.run(debug=True)

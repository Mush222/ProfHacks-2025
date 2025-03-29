import cv2
import numpy as np
from PIL import Image
import winsound
import time
   
def main():
    # Initialize video capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Load multiple face detection classifiers for different angles
    face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Multi-Angle Face Detection System Initialized")
    print("Press 'q' to quit")
    
    # Variables for beep control
    last_beep_time = 0
    beep_interval = 1.0  # Minimum time between beeps in seconds
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        faces_frontal = face_cascade_frontal.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Detect profile faces
        faces_profile = face_cascade_profile.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Also detect profile faces in the flipped image (for faces looking the other way)
        faces_profile_flipped = face_cascade_profile.detectMultiScale(
            cv2.flip(gray, 1),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Check if any face is detected
        face_detected = len(faces_frontal) > 0 or len(faces_profile) > 0 or len(faces_profile_flipped) > 0
        
        # Play beep if face detected and enough time has passed
        #current_time = time.time()
        #if face_detected and (distance <= 20):
        #    winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        #    last_beep_time = current_time
        
        # Draw appropriate boxes based on face orientation
        if len(faces_frontal) > 0:
            # Draw rectangles around detected frontal faces (blue)
            for (x, y, w, h) in faces_frontal:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Calculate distance based on face size
                distance = 100 - (w * 0.3)  # Simple estimation
                distance = max(0, min(100, distance))  # Clamp between 0 and 100
                cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if face_detected and (distance <= 20):
                    winsound.Beep(1000, 500) 

        elif len(faces_profile) > 0:
            # Draw rectangles around detected profile faces (green)
            for (x, y, w, h) in faces_profile:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Calculate distance based on face size
                distance = 100 - (w * 0.3)  # Simple estimation
                distance = max(0, min(100, distance))  # Clamp between 0 and 100
                cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if face_detected and (distance <= 20):
                    winsound.Beep(1000, 500)  
                           
        elif len(faces_profile_flipped) > 0:
            # Draw rectangles around detected flipped profile faces (red)
            for (x, y, w, h) in faces_profile_flipped:
                # Adjust x coordinate for flipped detection
                x = frame.shape[1] - x - w
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # Calculate distance based on face size
                distance = 100 - (w * 0.3)  # Simple estimation
                distance = max(0, min(100, distance))  # Clamp between 0 and 100
                cv2.putText(frame, f'Distance: {int(distance)}%', (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                if face_detected and (distance <= 20):
                    winsound.Beep(1000, 500) 
        
        # Display the frame
        cv2.imshow('Multi-Angle Face Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
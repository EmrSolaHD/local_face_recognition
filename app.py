"""
Face Recognition Application

This script performs real-time face recognition using a webcam. It loads known faces
from images in the 'known_faces' directory, detects faces in the video feed, and
identifies them by comparing against the known faces. Recognized faces are labeled
with their names, while unknown faces are marked as 'Unknown'.

Requirements:
- OpenCV (cv2)
- face_recognition library
- Images of known faces in 'known_faces/' folder (JPG or PNG format)

Usage:
Run the script to start the webcam feed. Press 'q' to quit.
"""

import cv2
import face_recognition
import os

# Initialize lists to store known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Directory containing images of known faces
known_faces_dir = "known_faces"

# Load and encode known faces from images in the directory
for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            # Use the filename (without extension) as the person's name
            known_face_names.append(os.path.splitext(filename)[0])

# Initialize webcam capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Starting video. Press 'q' to quit.")

# Main loop for real-time face recognition
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize frame to improve processing speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert from BGR (OpenCV default) to RGB (required by face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # List to hold names corresponding to detected faces
    face_names = []
    for face_encoding in face_encodings:
        # Compare current face encoding with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Calculate distances to find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if face_distances.size > 0:  # Ensure there are distances to compare
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # Draw rectangles and labels on the original frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back to original frame size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Choose color based on recognition: red for unknown, green for known
        if name == 'Unknown':
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (left-5, top - 25 - text_height - 5), (left + text_width + 5, top - 25 + 5), (128,128,128), -1)
            cv2.putText(frame, name, (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:    
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Get text size for background
            new_text = f"Hello {str(name).upper()} !"
            (text_width, text_height), baseline = cv2.getTextSize(new_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (left-5, top - 25 - text_height - 5), (left + text_width + 5, top - 25 + 5), (128,128,128), -1)
            cv2.putText(frame, new_text, (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the processed frame in a window
    cv2.imshow("Face Recognition", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()

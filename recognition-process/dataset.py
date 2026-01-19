import cv2
import pickle
import numpy as np
import os

# Initialize face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Video capture
video = cv2.VideoCapture(0)

# Create storage
faces_data = []
names = []

person_name = input("Enter your name: ")

# Collect 100 face samples
count = 0
while count < 100:
    ret, frame = video.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_crop, (50, 50)).flatten()

        faces_data.append(resized_face)
        names.append(person_name)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Collected: {count}/100", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Load existing data (if any) and append new data
if os.path.exists("data/faces_data.pkl") and os.path.exists("data/names.pkl"):
    with open("data/faces_data.pkl", "rb") as f:
        existing_faces = pickle.load(f)
    with open("data/names.pkl", "rb") as f:
        existing_names = pickle.load(f)
    
    faces_data = existing_faces + faces_data
    names = existing_names + names

# Save data
with open("data/faces_data.pkl", "wb") as f:
    pickle.dump(faces_data, f)

with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

print("Face dataset saved successfully!")



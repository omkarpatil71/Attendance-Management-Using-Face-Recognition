import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os

# Define paths
faces_path = "data/faces_data.pkl"
model_path = "data/svm_model.pkl"
scaler_path = "data/scaler.pkl"
today_date = datetime.now().strftime("%d-%m-%Y")
attendance_file = "Attendance/attendance"+today_date+".csv"

# Load trained model and scaler
with open(model_path, "rb") as f:
    svm_model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Load face data to map Student IDs to Names
with open(faces_path, "rb") as f:
    faces_data = pickle.load(f)

# Create a mapping of Student ID to Name
student_names = {sid: data["name"] for sid, data in faces_data.items()}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")


# Ensure attendance file exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Student ID", "Name", "Timestamp"])
    df.to_csv(attendance_file, index=False)

# Open webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    recognized_students = {}  # Temporary storage for detected faces

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)

        # Normalize face data
        resized_face = scaler.transform(resized_face)

        # Predict student ID
        sid_pred = svm_model.predict(resized_face)[0]

        # Get student name from ID
        student_name = student_names.get(sid_pred, "Unknown")

        # Store recognized students
        recognized_students[sid_pred] = student_name

        # Display results on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{student_name} ({sid_pred})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'O' to record attendance
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o') and recognized_students:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Read existing attendance data
        df = pd.read_csv(attendance_file)

        # Append attendance entries
        for sid, name in recognized_students.items():
            if name != "Unknown":
                new_entry = pd.DataFrame([[sid, name, timestamp]], columns=["Student ID", "Name", "Timestamp"])
                df = pd.concat([df, new_entry], ignore_index=True)
                print(f"✅ Attendance marked: {name} ({sid}) at {timestamp}")

        # Save updated attendance file
        df.to_csv(attendance_file, index=False)

    # Press 'Q' to exit
    if key == ord('q'):
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()

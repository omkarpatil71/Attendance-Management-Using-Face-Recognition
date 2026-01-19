from genericpath import exists
import cv2
import pickle
import numpy as np
import csv
from datetime import datetime
import time
import os

# Load face detector
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load trained model and scaler
with open('data/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load labels
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

# Start webcam
video = cv2.VideoCapture(0)

#creating two coloumns
COL_NAMES=['NAME','TIME']

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)

        # Normalize face
        resized_face = scaler.transform(resized_face)

        # Predict label
        output = svm_model.predict(resized_face)[0]

        #datetime specification
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        os.path.isfile("Attendance/Attendance_"+date+".csv")

        # Display prediction
        cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

        #stroring attendance
        attendance=[str(output[0:]),str(timestamp)]

    cv2.imshow("Face Recogonition", frame)

    if cv2.waitKey(1) & 0xFF == ord('o'):
        if exists("attendance.csv"):
            with open("Attendance/attendance_"+ date +".csv",'+a')as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/attendance_"+ date +".csv","+a")as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

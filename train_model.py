import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# Define paths
faces_path = "data/faces_data.pkl"
model_path = "data/svm_model.pkl"
scaler_path = "data/scaler.pkl"

# Check if face data file exists
if not os.path.exists(faces_path):
    raise FileNotFoundError("❌ Face data not found! Please add students first.")

# Load dataset
with open(faces_path, "rb") as f:
    faces_data = pickle.load(f)

# Extract features (faces) and labels (student IDs)
FACES = []
LABELS = []

for sid, data in faces_data.items():
    for face in data["faces"]:
        FACES.append(face)
        LABELS.append(sid)  # Store Student ID as label

# Convert to NumPy array
FACES = np.array(FACES)
LABELS = np.array(LABELS)

# Ensure at least two unique students exist for training
unique_labels = np.unique(LABELS)
if len(unique_labels) < 2:
    raise ValueError("❌ Dataset must contain at least two different students for training!")

# Normalize features
scaler = StandardScaler()
FACES = scaler.fit_transform(FACES)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(FACES, LABELS)

# Save model and scaler
with open(model_path, "wb") as f:
    pickle.dump(svm_model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model trained and saved successfully!")

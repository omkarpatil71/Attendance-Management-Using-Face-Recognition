from sklearn.svm import SVC
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Convert to NumPy array
FACES = np.array(FACES)
LABELS = np.array(LABELS)

# Ensure at least two different people exist
unique_labels = np.unique(LABELS)
if len(unique_labels) < 2:
    raise ValueError("Dataset must contain at least two different people!")

# Normalize features
scaler = StandardScaler()
FACES = scaler.fit_transform(FACES)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(FACES, LABELS)

# Save model and scaler
with open('data/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model trained and saved successfully!")

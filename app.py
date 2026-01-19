from flask import Flask,render_template,redirect,request,flash,url_for,session
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash,check_password_hash
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = "Omkar"

#database_connectivity
app.config["MONGO_URI"] = "mongodb+srv://root:toor@omkar.wuznt.mongodb.net/College"
mongo = PyMongo(app)

@app.route('/')
def home():
    # password="pass123"
    # hashed_password = generate_password_hash(password)
    # mongo.db.Admin_info.insert_one({"Admin_id": "admin123", "password": hashed_password})
    return render_template('index.html')

#admin login
@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    if request.method == "POST":
        Admin_id = request.form.get('Admin_id')  # Ensure correct input retrieval
        password = request.form.get('password')

        if not Admin_id or not password:
            flash("Both fields are required!", "danger")
            return render_template('adminlogin.html')

        # Fetch admin from DB
        Admin = mongo.db.Admin_info.find_one({"Admin_id": Admin_id})

        if Admin and check_password_hash(Admin["password"], password):
            session["Admin_id"] = str(Admin["Admin_id"])  # Ensure session stores string
            return redirect(url_for('admindashboard'))  # Use url_for() for proper routing

        flash("Invalid Admin ID or Password", "danger")
        return render_template('adminlogin.html')  # Ensure flash message is shown

    return render_template('adminlogin.html')  # Render login page for GET request

#admindashboard
@app.route('/admindashboard',methods=['GET','POST'])
def admindashboard():
     subject = request.args.get('subject')  # Get subject filter
     time_filter = request.args.get('time')  # Weekly or Monthly

    # Fetch attendance data
     query = {}
     if subject:
            query["subject"] = subject  # Filter by subject

     attendance_data = list(mongo.db.Attendance.find(query))
    
    # Convert timestamps to datetime
     for record in attendance_data:
        record["timestamp"] = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")

    # Filter by time range
     if time_filter == "weekly":
        start_date = datetime.now() - timedelta(days=7)
     elif time_filter == "monthly":
        start_date = datetime.now() - timedelta(days=30)
     else:
        start_date = None  # Show all data

     if start_date:
        attendance_data = [record for record in attendance_data if record["timestamp"] >= start_date]

    # Convert to DataFrame for easier manipulation
     df = pd.DataFrame(attendance_data)

    # Attendance count per student
     attendance_counts = df["student_id"].value_counts()

    # Generate Pie Chart
     fig, ax = plt.subplots(figsize=(4,4))
     ax.pie(attendance_counts, labels=attendance_counts.index, autopct='%1.1f%%', startangle=90)
     ax.axis('equal')

     ax.set_position([0.1, 0.1, 0.8, 0.8])
    # Convert plot to base64 image for HTML
     img = io.BytesIO()
     plt.savefig(img, format='png')
     img.seek(0)
     plot_url = base64.b64encode(img.getvalue()).decode()
     return render_template('admindashboard.html', students=attendance_data, plot_url=plot_url)

#adminlogout
@app.route('/adminlogout')
def adminlogout():
    session.pop("Admin_id", None)  # Remove student_id from session
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('adminlogin'))  # Redirect to login page

#student login
@app.route('/studentlogin', methods=['GET', 'POST'])
def studentlogin():
    if request.method == "POST":
        student_id = request.form.get('student_id')  # Fix tuple issue
        password = request.form.get('password')

        if not student_id or not password:
            flash("Both fields are required!", "danger")
            return render_template('studentlogin.html')

        # Fetch student from DB
        student = mongo.db.Student_info.find_one({"student_id": student_id})

        if student and check_password_hash(student["password"], password):
            session["student_id"] = str(student["student_id"])  # Ensure string
            session["student_name"] = student["student_name"]
            return redirect('/studentdashboard')

        flash("Invalid Student ID or Password", "danger")

    return render_template('studentlogin.html')

#reset password
@app.route('/resetpassword', methods=['GET', 'POST'])
def resetpassword():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        new_password = request.form.get('new_password')

        if not student_id or not new_password:
            flash("Student ID and new password are required!", "danger")
            return redirect(url_for('resetpassword'))

        student = mongo.db.Student_info.find_one({"student_id": student_id})

        if student:
            hashed_pass = generate_password_hash(new_password)  # Hash new password

            mongo.db.Student_info.update_one(
                {"student_id": student_id},
                {"$set": {"password": hashed_pass}}
            )
            flash("✅ Password changed successfully! Please log in.", "success")
            return redirect(url_for('studentlogin'))  # Redirect to login

        flash("❌ Student ID not found!", "danger")
        return redirect(url_for('resetpassword'))  # Redirect back

    return render_template('resetpassword.html')  # Render password reset page

#studentlogout
@app.route('/studentlogout')
def logout():
    session.pop("student_id", None)  # Remove student_id from session
    session.pop("student_name", None)  # Remove student_name if stored
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('studentlogin'))  # Redirect to login page

#studentdashboard
@app.route('/studentdashboard',methods=['GET','POST'])
def studentdashboard():
    if 'student_id' not in session:
        flash("Please log in to access your dashboard", "danger")
        return redirect(url_for('login'))  # Redirect if not logged in

    student_id = session['student_id']  # Get logged-in student's ID
    subject = request.args.get('subject')  # Get subject filter
    time_filter = request.args.get('time')  # Weekly or Monthly filter

    # Fetch attendance for the logged-in student
    query = {"student_id": student_id}
    if subject:
        query["subject"] = subject  # Filter by subject

    attendance_data = list(mongo.db.Attendance.find(query))
    
    # Convert timestamps to datetime
    for record in attendance_data:
        record["timestamp"] = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")

    # Filter by time range
    if time_filter == "weekly":
        start_date = datetime.now() - timedelta(days=7)
    elif time_filter == "monthly":
        start_date = datetime.now() - timedelta(days=30)
    else:
        start_date = None  # Show all data

    if start_date:
        attendance_data = [record for record in attendance_data if record["timestamp"] >= start_date]

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(attendance_data)

    # Attendance count per subject
    attendance_counts = df["subject"].value_counts()

    # Generate Pie Chart
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(attendance_counts, labels=attendance_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    # Convert plot to base64 image for HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('studentdashboard.html', attendance=attendance_data, plot_url=plot_url)
    

@app.route('/goto', methods=['GET','POST'])
def goto():
    return render_template('mark_attendance.html')

#studengt logiregistration
@app.route('/studentregistration',methods=['GET','POST'])
def studentregistration():
    if request.method == "POST":
        data = {
            "student_id": request.form['student_id'],
            "student_name": request.form['student_name'],
            "password": generate_password_hash(request.form['password'])  # Hash password
        }
        mongo.db.Student_info.insert_one(data)
    return render_template('studentregistration.html',)
   

#add new student
@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # Load existing face data
    faces_data = {}

    if os.path.exists("data/faces_data.pkl"):
        with open("data/faces_data.pkl", "rb") as f:
            faces_data = pickle.load(f)

    if request.method == 'POST':  # Start face capture only when form is submitted
        sname = request.form['sname']
        sid = request.form['sid']

        if not sid or not sname:
            return render_template("add_student.html")
        
        # Initialize dictionary if student ID is new
        if sid not in faces_data:
            faces_data[sid] = {"name": sname, "faces": []}

        video = cv2.VideoCapture(0)  # Open camera
        count = 0

        while count < 100:  # Collect 100 faces
            ret, frame = video.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_crop, (50, 50)).flatten()

                faces_data[sid]["faces"].append(resized_face)  # Store with Student ID
                count += 1

                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Collected: {count}/100", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Collecting Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release camera and close OpenCV windows
        video.release()
        cv2.destroyAllWindows()

        # Save updated face data
        with open("data/faces_data.pkl", "wb") as f:
            pickle.dump(faces_data, f)

        print(f"Face dataset for {sname} ({sid}) saved successfully!")
        return render_template('add_student.html', message="Face data saved successfully!")

    return render_template('add_student.html')  # Load form initially


#goto


#marking attendance
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    subject = request.form.get('subject')  # Get subject from form

    if not subject:
        flash("Please select a subject!", "danger")
        return redirect(url_for('goto'))

    today_date = datetime.now().strftime("%d-%m-%Y")
    attendance_file = f"Attendance/attendance_{today_date}.csv"

    # Ensure attendance file exists
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Student ID", "Name", "Subject", "Timestamp"])
        df.to_csv(attendance_file, index=False)

    # Load trained model and scaler
    with open("data/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)

    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load face data
    with open("data/faces_data.pkl", "rb") as f:
        faces_data = pickle.load(f)

    student_names = {sid: data["name"] for sid, data in faces_data.items()}
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

    video = cv2.VideoCapture(0)
    recognized_students = {}

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_crop, (50, 50)).flatten().reshape(1, -1)
            resized_face = scaler.transform(resized_face)

            sid_pred = str(svm_model.predict(resized_face)[0])
            student_name = student_names.get(sid_pred, "Unknown")

            if student_name != "Unknown":
                recognized_students[sid_pred] = student_name

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{student_name} ({sid_pred})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('o') and recognized_students:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            return redirect(url_for('goto'))

    video.release()
    cv2.destroyAllWindows()

    timestamp = datetime.now()
    df = pd.read_csv(attendance_file)

    for sid, name in recognized_students.items():
        last_attendance = mongo.db.Attendance.find_one(
            {"student_id": sid, "subject": subject},
            sort=[("timestamp", -1)]
        )

        if last_attendance:
            last_time = datetime.strptime(last_attendance["timestamp"], "%Y-%m-%d %H:%M:%S")
            if timestamp - last_time < timedelta(hours=1):
                flash(f"❌ {name} ({sid}) already marked attendance for {subject} in the last hour!", "danger")
                continue

        # Save to MongoDB
        attendance_record = {
            "student_id": sid,
            "name": name,
            "subject": subject,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        mongo.db.Attendance.insert_one(attendance_record)

        # Save to CSV
        new_entry = pd.DataFrame([[sid, name, subject, timestamp.strftime("%Y-%m-%d %H:%M:%S")]],
                                 columns=["Student ID", "Name", "Subject", "Timestamp"])
        df = pd.concat([df, new_entry], ignore_index=True)
        print(f"✅ Attendance marked: {name} ({sid}) for {subject} at {timestamp}")

    df.to_csv(attendance_file, index=False)

    flash("Attendance marked successfully!", "success")
    return redirect(url_for('goto'))



#training model
@app.route('/trainmodel',methods=["GET","POST"])
def trainmodel():
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
    flash("Model trained successfully","success")
    return render_template('trainmodel.html')



if __name__ == "__main__":
    app.run(debug=True,port=5000)
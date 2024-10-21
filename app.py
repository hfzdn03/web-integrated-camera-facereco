import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import date, datetime
import shutil
import face_recognition as fr
import pandas as pd
import threading
import numpy as np
import time
import base64
from werkzeug.utils import secure_filename

from datetime import date, datetime, timedelta

app = Flask(__name__)

nimgs = 7  # Reduce the number of images for face registration to 7
unknown_threshold = 0.5  # Adjusted threshold for recognizing unknown faces

# Placeholder variable for progress
progress = 0

# Create necessary directories if they do not exist
directories = [
    'static/faces/CNNalgo',
    'static/faces/MediaPipealgo',
    'static/faces/SVCalgo',
    'face_rec'
]

for directory in directories:
    if not os.path.isdir(directory):
        os.makedirs(directory)

def load_faces_and_update_progress():
    global loading_progress
    
    # Update known face encodings after adding a new user
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_face_encodings() 
    # Simulate progress for demonstration (replace with actual loading logic)
    for i in range(1, 101):
        loading_progress = i
        time.sleep(0.1)  # Simulate time taken for loading

# Create necessary directories and CSV file
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

attendance_csv = f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv'
if not os.path.isfile(attendance_csv):
    with open(attendance_csv, 'w') as f:
        f.write('Name,Roll,Time')

# Update the extract_attendance function to handle missing file
def extract_attendance():
    if not os.path.isfile(attendance_csv):
        return [], [], [], 0  # Return empty lists if file doesn't exist
    df = pd.read_csv(attendance_csv)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    if '_' not in name:
        return
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv')
    if int(userid) in list(df['Roll']):
        df.loc[df['Roll'] == int(userid), 'Time'] = current_time
    else:
        new_row = pd.DataFrame([[username, int(userid), current_time]], columns=['Name', 'Roll', 'Time'])
        df = pd.concat([df, new_row])
    
    df.to_csv(f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv', index=False)

# Global variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Load face encodings from the dataset
def load_face_encodings():
    known_encodings = []
    known_names = []
    userlist = os.listdir('face_rec')
    
    for user in userlist:
        for imgname in os.listdir(f'face_rec/{user}'):
            img_path = f'face_rec/{user}/{imgname}'
            image = fr.load_image_file(img_path)
            encoding = fr.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(user)
    return known_encodings, known_names

# Initialize a dictionary to store detection times
detection_start_time = {}

def process_frames(frame, face_data):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame to 1/4 size
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Face location detection
    face_locations = fr.face_locations(rgb_frame, model="hog")  
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]  # Scale back up
        name = identify_face(face_encoding)
        
        # Check if the face is not "Unknown" and record detection start time
        if name != "Unknown" and name not in detection_start_time:
            detection_start_time[name] = datetime.now()
        elif name != "Unknown":
            # Calculate detection duration
            detection_duration = datetime.now() - detection_start_time[name]
            if detection_duration >= MIN_DETECTION_DURATION:
                add_attendance(name)
                del detection_start_time[name]

        # Remove if detection exceeds timeout
        if name in detection_start_time and datetime.now() - detection_start_time[name] >= DETECTION_TIMEOUT:
            del detection_start_time[name]

        face_data.append((name, (left, top, right, bottom)))

def identify_face(face_encoding):
    """Identify the face from known encodings."""
    if len(known_face_encodings) == 0:
        return "Unknown"  # If no encodings exist, return Unknown immediately

    # Compare faces and calculate distances
    matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=unknown_threshold)
    face_distances = fr.face_distance(known_face_encodings, face_encoding)

    # If no matches or distances exist, return Unknown
    if len(face_distances) == 0 or not any(matches):
        return "Unknown"

    # Find the best match
    best_match_index = np.argmin(face_distances)

    # Ensure that the best match has a sufficiently low distance
    if matches[best_match_index] and face_distances[best_match_index] < unknown_threshold:
        return known_face_names[best_match_index]
    else:
        return "Unknown"

# Initialize a dictionary to store detection times
detection_start_time = {}

# Function to save images
def save_image(image, name, staff_id):
    # Create the directory path
    directory = f"face_rec/{name}_{staff_id}/"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a secure filename
    filename = secure_filename(image.filename)
    
    # Define the complete file path
    file_path = os.path.join(directory, filename)
    
    # Save the image
    image.save(file_path)
    return file_path

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=len(os.listdir('face_rec')), datetoday2=date.today().strftime("%d-%B-%Y"), algo='FaceReco mod', facereco = True)

@app.route('/start', methods=['GET'])
def start():
    """Start the face recognition process for detection."""
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_face_encodings()  # Load encodings at start
    cv2.namedWindow('Face Detection', cv2.WND_PROP_FULLSCREEN)

    face_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame, model="hog")
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        face_data.clear()  # Clear previous face data

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = identify_face(face_encoding)
            face_data.append((name, (left, top, right, bottom)))

            # Draw rectangles and labels
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Handle attendance check logic (if needed)
            if name != "Unknown" and name not in detection_start_time:
                detection_start_time[name] = datetime.now()
            elif name != "Unknown":
                detection_duration = datetime.now() - detection_start_time[name]
                if detection_duration >= timedelta(seconds=3):  # 3 seconds detection
                    # Add to attendance here if necessary
                    del detection_start_time[name]

        # Show the frame
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

@app.route('/add', methods=['POST'])
def add_user():
    """Add a new user and update face encodings"""
    newusername = request.form.get('staff_name')
    newuserid = request.form.get('staff_id')
    userimagefolder = f'face_rec/{newusername}_{newuserid}'

    # Create the user folder if it doesn't exist
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Get the images from the request
    images = request.files.getlist('image')  # Use getlist to handle multiple files

    # Save the images
    for i, image in enumerate(images):
        # Save each image using its filename to avoid overwriting
        filename = secure_filename(image.filename)  # Use secure_filename to prevent path issues
        image.save(os.path.join(userimagefolder, f"{filename}"))  # Save the image

    # Optionally process images for encoding here.

    return redirect('/')



@app.route('/users/', methods=['GET'])
def display_users():
    userlist = os.listdir('face_rec')
    return jsonify(userlist)

@app.route('/remove_user/<username>', methods=['POST'])
def remove_user(username):
    user_folder = os.path.join('face_rec', username)
    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)
        return jsonify({'message': f'Successfully removed {username}'}), 200
    return jsonify({'message': 'User not found'}), 404

@app.route('/reload', methods=['GET'])
def reload():
    global progress
    progress = 0  # Reset progress
    # Simulate reloading process
    for _ in range(10):
        time.sleep(1)  # Simulating time taken for each step
        progress += 10  # Increment progress
    return jsonify({"message": "Reloading started"}), 200

@app.route('/reload_progress', methods=['GET'])
def reload_progress():
    return jsonify({"progress": progress}), 200

if __name__ == '__main__':
    app.run(debug=True)
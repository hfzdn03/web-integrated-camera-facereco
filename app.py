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
import face_recognition
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import pytz

from datetime import date, datetime, timedelta

app = Flask(__name__)

nimgs = 7  # Reduce the number of images for face registration to 7
unknown_threshold = 0.5  # Adjusted threshold for recognizing unknown faces

# Placeholder variable for progress
progress = 0

is_reloading = False  # Initially, reloading is not happening

# Create necessary directories if they do not exist
directories = [
    'static/faces/CNNalgo',
    'static/faces/MediaPipealgo',
    'static/faces/SVCalgo',
    'face_rec'
]

# Define the minimum detection duration and detection timeout
MIN_DETECTION_DURATION = timedelta(seconds=2)  # 2 seconds
DETECTION_TIMEOUT = timedelta(seconds=1)  # 1 seconds

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

    # Set the timezone to Kuala Lumpur/Singapore
    kl_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    current_time = datetime.now(kl_timezone).strftime("%H:%M:%S")
    
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
                # Safely delete the name from detection_start_time if it exists
                if name in detection_start_time:
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

@app.route('/add', methods=['POST'])
def add_user():
    """Add a new user and update face encodings"""
    newusername = request.form.get('staff_name')
    newuserid = request.form.get('staff_id')
    userimagefolder = f'face_rec/{newusername}_{newuserid}'

    # Create the user folder if it doesn't exist
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    # Get the image data from the request
    img_data = request.form.get('image')  # Get base64-encoded image

    # Decode the base64 image
    img_data = img_data.split(',')[1]  # Remove the data URL header
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))  # Convert to PIL Image

    # Convert the image to RGB (to handle cases where the image might be RGBA or other formats)
    img = img.convert('RGB')

    # Convert image to NumPy array for face detection
    img_array = np.array(img)

    # Detect face locations
    face_locations = face_recognition.face_locations(img_array)

    if face_locations:
        # Extract the first detected face
        top, right, bottom, left = face_locations[0]

        # Crop the face region
        face_image = img.crop((left, top, right, bottom))

        # Save the cropped face
        face_image.save(os.path.join(userimagefolder, f"{newusername}_{newuserid}.png"))
    else:
        return "No face detected in the image.", 400

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
    """Reload the dataset and pause face processing while reloading."""
    global is_reloading, progress
    is_reloading = True  # Set reloading flag to true
    progress = 0  # Reset progress
    
    # Function to simulate the reloading process with progress updates
    def reload_data():
        global known_face_encodings, known_face_names, progress, is_reloading
        try:
            # Reload the face encodings
            known_face_encodings, known_face_names = load_face_encodings()

            # Simulating time-consuming reloading process
            for _ in range(10):
                time.sleep(1)  # Simulate each reloading step
                progress += 10  # Increment progress

        finally:
            # Ensure the reloading flag is reset even if an error occurs
            is_reloading = False

    # Start the reloading process in a separate thread so it doesn't block the app
    threading.Thread(target=reload_data).start()

    return jsonify({"message": "Reloading started"}), 200

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global is_reloading

    # If reloading is in progress, wait until it finishes
    while is_reloading:
        time.sleep(0.1)  # Wait briefly before checking again

    # Continue with face processing once reloading is done
    data = request.form.get('image')
    
    # Convert base64 to image
    image_data = base64.b64decode(data.split(',')[1])
    img = Image.open(BytesIO(image_data))
    img = img.convert('RGB')  # Ensure it's RGB
    frame = np.array(img)

    # Process the frame
    face_data = []
    process_frames(frame, face_data)

    # Return face locations and labels to front end
    return jsonify({"face_data": [{"name": name, "box": box} for name, box in face_data]})

@app.route('/reload_progress', methods=['GET'])
def reload_progress():
    """Get the current progress of the reloading process."""
    global progress, is_reloading
    return jsonify({
        "progress": progress,
        "is_reloading": is_reloading
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
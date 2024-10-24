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

# Track the last detection time for both time in and time out
last_detection = {}  # Will store the last detection time for each user

TIMEOUT_WINDOW = timedelta(minutes=3)  # 3 minutes window for both in/out

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

# Create necessary directories and CSV file
attendance_csv = f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv'
try:
    if not os.path.isfile(attendance_csv):
        with open(attendance_csv, 'w') as f:
            f.write('Name,Roll,Time In,Time Out')
except Exception as e:
    print(f"Error creating CSV file: {e}")

def extract_attendance():
    if not os.path.isfile(attendance_csv):
        return [], [], [], [], 0  # Return empty lists if file doesn't exist
    df = pd.read_csv(attendance_csv)
    
    # Replace NaN values with an empty string
    df.fillna('', inplace=True)
    
    names = df['Name']
    rolls = df['Roll']
    times_in = df['Time In']
    times_out = df['Time Out']
    l = len(df)
    return names, rolls, times_in, times_out, l

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

def add_attendance(name, action):
    if '_' not in name:
        return
    username, userid = name.split('_')

    # Set the timezone to Kuala Lumpur/Singapore
    kl_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    current_time = datetime.now(kl_timezone)
    current_time_str = current_time.strftime("%H:%M:%S")
    
    # Load the attendance CSV for today
    attendance_file = f'Attendance/FaceReco-{date.today().strftime("%m_%d_%y")}.csv'
    
    if not os.path.isfile(attendance_file):
        # If the CSV doesn't exist, create it with headers
        df = pd.DataFrame(columns=['Name', 'Roll', 'Time In', 'Time Out'])
    else:
        # Load the existing CSV file
        df = pd.read_csv(attendance_file)

    # Check if the user has already checked in for the day
    user_records = df[df['Roll'] == int(userid)]

    if action == 'in':
        # Log Time In if no previous record exists or last record has Time Out
        if user_records.empty or pd.notna(user_records.iloc[-1]['Time Out']):
            new_row = pd.DataFrame([[username, int(userid), current_time_str, '']], 
                                   columns=['Name', 'Roll', 'Time In', 'Time Out'])
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"Logged Time In for {username} at {current_time_str}")
    elif action == 'out':
        # Log Time Out only if there is a Time In that hasn't been followed by a Time Out
        if not user_records.empty and pd.isna(user_records.iloc[-1]['Time Out']):
            df.loc[df.index == user_records.index[-1], 'Time Out'] = current_time_str
            print(f"Logged Time Out for {username} at {current_time_str}")

    # Save the updated CSV back to the file
    df.to_csv(attendance_file, index=False)

def process_frames(frame, face_data):
    global last_detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Resize frame to 1/4 size
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    current_time = datetime.now()

    # Face location detection
    face_locations = fr.face_locations(rgb_frame, model="hog")  
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]  # Scale back up
        name = identify_face(face_encoding)

        # Only process known faces
        if name != "Unknown":
            if name not in last_detection:
                last_detection[name] = {'last_time_in': None, 'last_time_out': None}

            # Print the current state of last_detection
            print(f"Current state for {name} before processing: {last_detection[name]}")
            
            # Check if the user should be marked for Time In
            if (last_detection[name]['last_time_in'] is None or 
                (current_time - last_detection[name]['last_time_in']) >= TIMEOUT_WINDOW):
                
                # Allow a new Time In if the last Time Out was logged
                if last_detection[name]['last_time_out'] is not None:
                    last_detection[name]['last_time_in'] = None  # Reset last_time_in after Time Out

                if last_detection[name]['last_time_in'] is None:  # Only log if not already logged
                    add_attendance(name, 'in')  # Mark attendance for Time In
                    last_detection[name]['last_time_in'] = current_time
                    print(f"Logged Time In for {name} at {current_time}")

            # Check if the user should be marked for Time Out
            if last_detection[name]['last_time_out'] is None and last_detection[name]['last_time_in'] is not None:
                # Check if the timeout window has passed since last Time In
                if (current_time - last_detection[name]['last_time_in']) >= TIMEOUT_WINDOW:
                    add_attendance(name, 'out')  # Mark attendance for Time Out
                    last_detection[name]['last_time_out'] = current_time
                    print(f"Logged Time Out for {name} at {current_time}")
                else:
                    print(f"{name} has not yet timed out. Time since last Time In: {current_time - last_detection[name]['last_time_in']}")
            
            # Print the time since last Time In (for debugging)
            print(f"{name} - Time since last Time In: {current_time - last_detection[name]['last_time_in']}")
        
        # Append face data regardless of whether it's known or unknown
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
    names, rolls, times_in, times_out, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times_in=times_in, times_out=times_out, l=l, totalreg=len(os.listdir('face_rec')), datetoday2=date.today().strftime("%d-%B-%Y"), algo='FaceReco mod', facereco = True)

@app.route('/attendance_data', methods=['GET'])
def attendance_data():
    names, rolls, times_in, times_out, l = extract_attendance()
    return jsonify({
        'names': names.tolist(),
        'rolls': rolls.tolist(),
        'times_in': times_in.tolist(),
        'times_out': times_out.tolist(),
        'length': l
    })

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
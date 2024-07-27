from random import randint
from time import time
from flask import Flask, Response, jsonify, render_template, request
import face_recognition
import cv2
import os
import json
import numpy as np

app = Flask(__name__)

SAVED_FACES = {}
SAVED_FACES_FILE = 'data/encoding_list.json'
devices = []
# save face encodings to np file
def save_face_encoding(student_id, encoding):
    global SAVED_FACES
    
    # Update or add the new encoding
    SAVED_FACES[student_id] = encoding.tolist()  # Convert to list for JSON serialization
    
    # Save the updated encodings
    os.makedirs('data', exist_ok=True)
    try:
        with open(SAVED_FACES_FILE, 'w') as f:
            json.dump(SAVED_FACES, f)
        print(f'Encoding for student ID {student_id} {"updated" if student_id in SAVED_FACES else "added"}')
    except Exception as e:
        print(f"Error saving encodings: {e}")


# retrieve encodings from file
def load_saved_encodings():
    global SAVED_FACES
    if os.path.exists(SAVED_FACES_FILE):
        try:
            with open(SAVED_FACES_FILE, 'r') as f:
                file_content = f.read().strip()
                if file_content:  # Check if the file is not empty
                    SAVED_FACES = json.loads(file_content)
                    # Convert lists back to numpy arrays
                    for student_id, encoding in SAVED_FACES.items():
                        SAVED_FACES[student_id] = np.array(encoding)
                    print('Encodings loaded')
                else:
                    print('Encoding file is empty, initializing with an empty dictionary')
                    SAVED_FACES = {}
        except json.JSONDecodeError:
            print('Error decoding JSON, initializing with an empty dictionary')
            SAVED_FACES = {}
    else:
        print('Encoding file does not exist, initializing with an empty dictionary')
        SAVED_FACES = {}

### Registers a new face
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    filename, extension = os.path.splitext(file.filename)
    new_filename = f'imgs/{int(time())}{extension}'

    # student_id = request.form.get('student_id')
    student_id = randint(2, 100)
    if not student_id:
        return jsonify({'success': False, 'message': 'No student ID provided'})

    os.makedirs('imgs', exist_ok=True)

    try:
        file.save(new_filename)
        image = face_recognition.load_image_file(new_filename)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            print(encoding)
            save_face_encoding(student_id, encoding)
            return jsonify({'status': 'success', 'message': 'Face registered successfully', 'encodings': str(encoding.tolist())}), 200
        else:
            return jsonify({'status': 'error', 'message': 'No face found in the image'}), 404

    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Recognizes student from the image passed in the request
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        encoding = encodings[0]
        matches = face_recognition.compare_faces(list(SAVED_FACES.values()), encoding)
        if True in matches:
            matched_idx = matches.index(True)
            matched_name = list(SAVED_FACES.keys())[matched_idx]
            return jsonify({'status': 'success', 'student_id': matched_name})
        else:
            return jsonify({'status': 'error', 'message': 'No match found'})
    else:
        return jsonify({'status': 'error', 'message': 'No face found in the image'})


# home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# generate frames from video
def generate_frames():
    # check for external cameras and use as default
    if len(devices) > 1:
        cam = 1
    else:
        cam = 0

    # initial camera with defautl device
    # camera = cv2.VideoCapture(devices[1])
    camera = cv2.VideoCapture(1)
    while True:
        success, frame = camera.read()
        if not success:
            camera.release()
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes() 
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Function to get a list of available camera devices
def list_camera_devices(max_devices=10):
    available_devices = []
    for device in range(max_devices):
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            available_devices.append(device)
            cap.release()
    return available_devices

# Get the list of available camera devices

# Print the list of devices
print(f"Available camera devices: {len(devices)}")

# get the video stream
@app.route("/stream")
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Capture a single frame and recognize face
@app.route("/recognize_from_camera", methods=['POST'])
def recognize_from_camera():
    if len(devices) == 0:
        return jsonify({'status': 'error', 'message': 'No camera devices found'})

    if len(devices) >= 1:
        cam = 1
    else:
        cam = 0

    camera = cv2.VideoCapture(1)
    success, frame = camera.read()
    camera.release()

    if not success:
        return jsonify({'status': 'error', 'message': 'Failed to capture image from camera'})

    # Convert the frame to RGB format (required by face_recognition)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)

    if len(encodings) > 0:
        encoding = encodings[0]
        known_encodings = list(SAVED_FACES.values())
        print(known_encodings)
        matches = face_recognition.compare_faces(list(SAVED_FACES.values()), encoding)
        if True in matches:
            matched_idx = matches.index(True)
            matched_name = list(SAVED_FACES.keys())[matched_idx]
            return jsonify({'status': 'success', 'student_id': matched_name, 'encoding': str(encoding)})
        else:
            return jsonify({'status': 'error', 'message': 'No match found', 'encoding': str(encoding)})
    else:
        return jsonify({'status': 'error', 'message': 'No face found in the image', 'encoding': str(encoding)})



@app.route("/session", methods=['GET'])
def startSession():
    return render_template('session.html')

# main 
if __name__ == '__main__':
    try:
        devices = list_camera_devices()
    except:
        print("Error getting camera devices")
    # os.makedirs('/data', exist_ok=True, mode=777)

    load_saved_encodings()
    app.run(port=3000, debug=True)


from random import randint
from time import time
from flask import Flask, Response, jsonify, render_template, request
import face_recognition
import cv2
import os
import json
import numpy as np
from flask_mysqldb import MySQL


app = Flask(__name__)

SAVED_FACES = {}
SAVED_FACES_FILE = 'data/encoding_list.json'
devices = []
mysql = MySQL(app)
print('Initializing database')
# Required
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "Costero23#"
app.config["MYSQL_DB"] = "attendance_system_db"


# init a connection
def get_db_connection():
    try:
        conn = mysql.connect()
        cursor = conn.cursor()
        return cursor
    except Exception as e:
        print(e)
        return None

# define a general query function
def query_db(query, args=None):
    cursor = get_db_connection()
    if cursor is None:
        return None
    try:
        cursor.execute(query, args or ())
        result = cursor.fetchall()
        cursor.close()
        return result
    except Exception as e:
        print(e)
        return None

# save face encodings to the database
def save_face_encoding(student_id, encoding):
    try:
        cursor = get_db_connection()
        if cursor is None:
            return {'status': 'error', 'message': 'Database connection failed'}

        # Check if the student ID already exists
        query = "SELECT COUNT(*) FROM recognitionss WHERE student_id = %s"
        cursor.execute(query, (student_id,))
        exists = cursor.fetchone()[0]

        if exists:
            query = "UPDATE recognitionss SET face_encoding = %s WHERE student_id = %s"
        else:
            query = "INSERT INTO recognitionss (student_id, face_encoding) VALUES (%s, %s)"

        cursor.execute(query, (student_id, json.dumps(encoding.tolist())))
        mysql.connection.commit()
        cursor.close()
        return {'status': 'success', 'message': 'Encoding saved'}
    except Exception as e:
        print(f"Error saving encodings: {e}")
        return {'status': 'error', 'message': str(e)}

# retrieve encodings from database
def load_saved_encodings():
    try:
        encodings = query_db("SELECT student_id, face_encoding FROM recognitionss")
        saved_faces = {}
        for student_id, encoding in encodings:
            saved_faces[student_id] = np.array(json.loads(encoding))
        print('Encodings loaded')
        return saved_faces
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return {}

def load_find():
    res = query_db("SELECT Concat (student_id, ':', face_encoding) FROM recognitionss")
    print(jsonify(res))
    return jsonify(res)

### Registers a new face
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    filename, extension = os.path.splitext(file.filename)
    new_filename = f'imgs/{int(time())}{extension}'

    student_id = randint(20000000, 100000000)
    if not student_id:
        return jsonify({'success': False, 'message': 'No student ID provided'})

    os.makedirs('imgs', exist_ok=True)

    try:
        file.save(new_filename)
        image = face_recognition.load_image_file(new_filename)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            save_response = save_face_encoding(student_id, encoding)
            if save_response['status'] == 'success':
                return jsonify({'status': 'success', 'message': 'Face registered successfully', 'encodings': str(encoding.tolist())}), 200
            else:
                return jsonify({'status': 'error', 'message': save_response['message']}), 500
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
        saved_faces = load_saved_encodings()
        matches = face_recognition.compare_faces(list(saved_faces.values()), encoding)
        if True in matches:
            matched_idx = matches.index(True)
            matched_name = list(saved_faces.keys())[matched_idx]
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
    cam = 1

    # initial camera with defautl device
    # camera = cv2.VideoCapture(devices[1])
    try:
        camera = cv2.VideoCapture(1)
    except: 
        camera = cv2.VideoCapture(0)

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
    # if len(devices) == 0:
    #     return jsonify({'status': 'error', 'message': 'No camera devices found'})

    # if len(devices) >= 1:
    #     cam = 1
    # else:
    #     cam = 0

    # camera = cv2.VideoCapture(0)
    # success, frame = camera.read()
    # camera.release()

    # if not success:
    #     return jsonify({'status': 'error', 'message': 'Failed to capture image from camera'})

    if not 'image' in request.files:
        return jsonify({'status': 'error', 'message': 'Failed to capture image from camera'})

    frame = request.files['image']
    
    # Convert the frame to RGB format (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)

    if len(encodings) > 0:
        encoding = encodings[0]
        saved_faces = load_saved_encodings()
        matches = face_recognition.compare_faces(list(saved_faces.values()), encoding)
        if True in matches:
            matched_idx = matches.index(True)
            matched_name = list(saved_faces.keys())[matched_idx]
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
        load_saved_encodings()
    except:
        print("Error getting camera devices")
    # os.makedirs('/data', exist_ok=True, mode=777)

    app.run(port=3000)
   
    # load_and_compare()


# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from time import time
from PIL import Image
from click import open_file
from flask import Flask, Response, jsonify, render_template, request
import face_recognition
import cv2
import os
import numpy as np

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# # the associated function.


name = "Kweku AIkins"

known_faces = {}

SAVED_FACES = {}

SAVED_FACES_FILE = 'encoding_list.npy'

def save_face_encoding(student_id, encoding):
    # np.save(os.path.join(SAVED_FACES_FILE, ), encoding)
    with open(SAVED_FACES_FILE, 'w+') as save_file:
        temp = SAVED_FACES.copy()
        SAVED_FACES[student_id] = encoding
        save_file.write(str(SAVED_FACES))
        print('Encodings saved')

def get_saved_encodings():
    with open(SAVED_FACES_FILE, 'w+') as faces:
        print(faces.read())

### registers a new face
# Temporal. Move to backend instead
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No Image uploaded'})

    file = request.files['image']
    # Generate a unique filename with extension
    filename, extension = os.path.splitext(file.filename)
    new_filename = f'imgs/{time}{extension}'

    print(request.form.get('student_id'))

    # Create the 'imgs' folder if it doesn't exist
    os.makedirs('imgs', exist_ok=True)  # Handle potential path issues

    try:
        # Save the uploaded image with the generated filename
        file.save(new_filename)

        # file = request.files['image']
        # img = "aikins.jpeg"
        
        image = face_recognition.load_image_file(new_filename)
        # image = face_recognition.load_image_file("aikins.jpeg")
        encodings = face_recognition.face_encodings(image)
        # print(encodings[0])
        if len(encodings) > 0:
            encoding = encodings[0]
            save_face_encoding(name, encoding)
            print(SAVED_FACES)
            return jsonify({'status': 'success', 'message': 'Face registered successfully', 'encodings' : str(encoding)}), 200
        else:
            return jsonify({'status': 'error', 'message': 'No face found in the image'}), 404 

    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# Recognized student from the image passed in the request
@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['image']
    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        encoding = encodings[0]
        matches = face_recognition.compare_faces(list(known_faces.values()), encoding)
        if True in matches:
            matched_idx = matches.index(True)
            matched_name = list(known_faces.keys())[matched_idx]
            return jsonify({'status': 'success', 'name': matched_name})
        else:
            return jsonify({'status': 'error', 'message': 'No match found'})
    else:
        return jsonify({'status': 'error', 'message': 'No face found in the image'})


@app.route('/test', methods=['POST'])
def test():
    # return request.files['image'].filename
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Missing image file.'}), 400

    try:
        encodings = face_recognition.face_encodings(face_recognition.load_image_file(request.files['image']))
        return jsonify({'status': 'success', 'message': 'Face found', 'encoding_data' : encodings[0].tolist()}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return render_template('register.html')

# read frames from camera as a live stream
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:

        # read frames from cam
        success, frame=camera.read()
        if not success: 
            camera.release()
            break
        else: 
            ret,buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			frame + b'\r\n')

@app.route("/stream")
def stream(): 
    """
    register a new students's face 
    """  
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(port=3000, debug=True,)



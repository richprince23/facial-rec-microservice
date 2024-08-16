from random import randint
from time import time
from flask import Flask, Response, jsonify, render_template, request
import face_recognition
import os
import json
import numpy as np
from flask_mysqldb import MySQL
from flask_cors import CORS
from config import Config


app = Flask(__name__)


cors = CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:8000",  
        # "allow_headers": "Authorization, Content-Type, x-csrf-token"
    }
})

app.config.from_object(Config)

devices = []

mysql = MySQL(app)


# init a connection
def get_db_connection():
    try:
        conn = mysql.connection
        cursor = conn.cursor()
        return cursor
    except Exception as e:
        print(f"Database connection error: {e}")
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
        print(f"Query error: {e}")
        return None

def load_saved_encodings():
    try:
        encodings = query_db("SELECT student_id, face_encoding FROM recognitions")
        if encodings is None:
            return {}
        print(str(encodings))
        saved_faces = {}
        for row in encodings:
            student_id, encoding = row
            try:
                # Ensure the encoding is a string before parsing
                if isinstance(encoding, str):
                    saved_faces[student_id] = np.array(json.loads(encoding))
                else:
                    print(f"Invalid encoding format for student_id {student_id}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error for student_id {student_id}: {e}")
                continue
        print('Encodings loaded')
        return saved_faces
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return {}


### Registers a new face
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded'})

    file = request.files['image']
    # filename, extension = os.path.splitext(file.filename)
    # new_filename = f'imgs/{int(time())}{extension}'

    student_id = randint(20000000, 100000000)
    if not student_id:
        return jsonify({'success': False, 'message': 'No student ID provided'})

    os.makedirs('imgs', exist_ok=True)

    try:
        # file.save(new_filename)
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)

        file = None
        if len(encodings) > 0:
            encoding = encodings[0]
            # save_response = save_face_encoding(student_id, encoding)
            # if save_response['status'] == 'success':
            return jsonify({'status': 'success', 'message': 'Face registered successfully', 'encodings': str(encoding.tolist())}), 200
            # else:
            #     return jsonify({'status': 'error', 'message': save_response['message']}), 500
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

        if not saved_faces:
            return jsonify({'status': 'error', 'message': 'No saved faces found'})

        # Calculate distances between the input encoding and all saved encodings
        face_distances = face_recognition.face_distance(list(saved_faces.values()), encoding)
        
        # Find the index of the minimum distance
        best_match_index = np.argmin(face_distances)
        best_match_distance = face_distances[best_match_index]

        # Define a stricter threshold for what is considered a match (lower values are better)
        if best_match_distance < 0.5:  # Adjusted threshold
            matched_name = list(saved_faces.keys())[best_match_index]
            return jsonify({'status': 'success', 'student_id': matched_name, 'confidence': 1 - best_match_distance})
        else:
            return jsonify({'status': 'error', 'message': 'No match found', 'distance': float(best_match_distance)})
    else:
        return jsonify({'status': 'error', 'message': 'No face found in the image'})

# main 
if __name__ == '__main__':
    app.run(port=3000)

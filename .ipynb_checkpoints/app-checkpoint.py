# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

from copy import copy
from random import randint, random
import shutil
import tempfile
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
FACE_REGISTRY_PATH = 'known_faces/'

def save_face_encoding(name, encoding):
    np.save(os.path.join(FACE_REGISTRY_PATH, name + '.npy'), encoding)

def load_known_faces():
    for filename in os.listdir(FACE_REGISTRY_PATH):
        if filename.endswith('.npy'):
            name = filename[:-4]
            encoding = np.load(os.path.join(FACE_REGISTRY_PATH, filename))
            known_faces[name] = encoding

def convert_to_rgb(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB
        rgb_img = img.convert('RGB')
        # Convert to 8-bit depth
        rgb_img = rgb_img.point(lambda p: p * 256)
        return np.array(rgb_img)
    
def remove_alpha(image_path):
    with Image.open(image_path) as img:
        # Remove alpha channel if it exists
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        return np.array(img)

def check_image_format_and_depth(image_path):

    # Open the image with Pillow to detect mode
    image = Image.open(image_path)
    mode = image.mode
    
    # Get the bit depth
    if hasattr(image, 'bits'):
        bit_depth = image.bits
    elif mode in ['RGB', 'RGBA']:
        bit_depth = 8 * len(mode)  # 24 for RGB, 32 for RGBA
    elif mode == 'L':
        bit_depth = 8  # 8-bit for grayscale
    else:
        bit_depth = 'Unknown'

    # If the image is in RGB or RGBA mode, we can use OpenCV to check BGR/RGB
    if mode in ["RGB", "RGBA"]:
        # Convert the image to a numpy array
        image_np = np.array(image)
        # Check the format by examining the first pixel
        first_pixel = image_np[0, 0]
        if len(first_pixel) >= 3 and first_pixel[0] > first_pixel[2]:  # Blue value is higher than Red value
            color_format = "BGR"
        else:
            color_format = "RGB"
        return f'{mode} ({color_format}), {bit_depth}-bit'
    else:
        return f'{mode}, {bit_depth}-bit'

def prepare_image_for_face_recognition(image_path):
    # Read the image with OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB (OpenCV uses BGR by default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ensure the image is 8-bit
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    
    return rgb_image

### registers a new face
# Temporal. Move to backend instead
@app.route('/register', methods=['POST'])
def register():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No Image uploaded'})

    # file = request.files['image']
    # # Generate a unique filename with extension
    # filename, extension = os.path.splitext(file.filename)
    # new_filename = f'imgs/{randint(1, 10000)}{extension}'

    # # Create the 'imgs' folder if it doesn't exist
    # os.makedirs('imgs', exist_ok=True)  # Handle potential path issues

    try:
        # Save the uploaded image with the generated filename
        # file.save(new_filename)

        file = request.files['image']
        
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            encoding = encodings[0]
            # name = request.form['name']
            # save_face_encoding(name, encoding)
            return jsonify({'status': 'success', 'message': 'Face registered successfully', 'encodings' : encoding})
    # else:
    #     return jsonify({'status': 'error', 'message': 'No face found in the image'})

    except Exception as e:
        print(f"Error during image processing: {e}")
        return jsonify({'success': False, 'message': str(e)})


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
        return jsonify({'status': 'success', 'message': 'Face found', 'encoding_data' : encodings[0].tolist()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/trya', methods=['GET'])
def trya():
    return jsonify({'message': 'trying a new route'})

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
    app.run()


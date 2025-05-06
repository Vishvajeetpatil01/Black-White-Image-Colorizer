from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import matplotlib.pyplot as plt
from colorizers import *
import torch
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import socket



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for mobile access
host_ip = socket.gethostbyname(socket.gethostname())

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)

# Load the colorization models
try:
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
except Exception as e:
    print(f"Error loading colorizers: {e}")
    exit()

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        img = load_img(img_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        eccv16_filename = f"{filename}_eccv16.png"
        siggraph17_filename = f"{filename}_siggraph17.png"

        eccv16_path = os.path.join(app.config['PROCESSED_FOLDER'], eccv16_filename)
        siggraph17_path = os.path.join(app.config['PROCESSED_FOLDER'], siggraph17_filename)

        plt.imsave(eccv16_path, out_img_eccv16)
        plt.imsave(siggraph17_path, out_img_siggraph17)

        return jsonify({
            'original': f'http://{host_ip}:5000/uploads/{filename}',
            'eccv16': f'http://{host_ip}:5000/static/{eccv16_filename}',
            'siggraph17': f'http://{host_ip}:5000/static/{siggraph17_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_filter', methods=['GET'])
def apply_filter():
    try:
        image_path = request.args.get('image')
        filter_type = request.args.get('filter')

        if not image_path or not filter_type:
            return jsonify({'error': 'Missing image or filter parameter'}), 400

        # Load the image
        full_image_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
        image = cv2.imread(full_image_path)

        if image is None:
            return jsonify({'error': 'Image could not be loaded'}), 500

        # Apply filters
        if filter_type == 'light':
            image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        elif filter_type == 'effect':
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            image = cv2.filter2D(image, -1, kernel)
        elif filter_type == 'magic':
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        elif filter_type == 'color_toning':
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return jsonify({'error': 'Invalid filter type'}), 400

        # Ensure the image is not cropped before saving
        print(f"Filtered image shape: {image.shape}")

        # Save filtered image
        filter_filename = f"{os.path.basename(image_path).split('.')[0]}_{filter_type}.png"
        filter_path = os.path.join(app.config['PROCESSED_FOLDER'], filter_filename)
        cv2.imwrite(filter_path, image)

        if not os.path.exists(filter_path):
            return jsonify({'error': 'Filtered image not saved correctly'}), 500

        return jsonify({'filtered_image': f'http://localhost:5000/static/{filter_filename}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


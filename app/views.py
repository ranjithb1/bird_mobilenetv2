from app import app
from flask import request, render_template, flash, redirect, url_for
import numpy as np
from PIL import Image
import string
import random
import os
import tflite_runtime.interpreter as tflite
from werkzeug.utils import secure_filename
import time
import csv
from datetime import datetime
import traceback
from pathlib import Path

# Configuration
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'heic'}
app.config['MAX_FILE_SIZE'] = 5 * 1024 * 1024  # 5MB

# Create directories
os.makedirs(app.config['INITIAL_FILE_UPLOADS'], exist_ok=True)
os.makedirs("app/static/logs", exist_ok=True)

# CSV log setup
csv_log_path = "app/static/logs/inference_log.csv"
if not os.path.isfile(csv_log_path):
    with open(csv_log_path, mode='w', newline='') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            "Timestamp", "Filename", "Predicted Class", "Confidence (%)",
            "Preprocessing Time (ms)", "Inference Time (ms)", "Total Time (ms)"
        ])

# --- MODEL LOADING ---
try:
    # Get absolute path to model
    current_dir = Path(__file__).parent
    model_path = current_dir / 'static' / 'model' / 'mobilenetv2_quant.tflite'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=str(model_path))
   
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    app.logger.info("TFLite model loaded successfully")
    app.logger.info(f"Input details: {input_details}")
    app.logger.info(f"Output details: {output_details}")

except Exception as e:
    app.logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Failed to load model - check logs") from e

# Class labels
classes = [
    'AMERICAN GOLDFINCH',
    'BARN OWL',
    'CARMINE BEE-EATER',
    'DOWNY WOODPECKER',
    'EMPEROR PENGUIN',
    'FLAMINGO'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", full_filename='images/white_bg.jpg')

    if request.method == "POST":
        if 'image' not in request.files:
            flash('No file uploaded!', 'error')
            return redirect(request.url)

        image_upload = request.files['image']

        if image_upload.filename == '':
            flash('No selected file!', 'error')
            return redirect(request.url)

        if not allowed_file(image_upload.filename):
            flash('Invalid file type! Allowed: PNG, JPG, JPEG, HEIC.', 'error')
            return redirect(request.url)

        try:
            # Generate unique filename
            file_ext = os.path.splitext(image_upload.filename)[1]
            name = ''.join(random.choices(string.ascii_lowercase, k=10)) + file_ext
            full_filename = 'uploads/' + name
            image_path = os.path.join(app.config['INITIAL_FILE_UPLOADS'], name)

            start_total = time.time()

            # --- PREPROCESSING ---
            start_pre = time.time()
            try:
                image = Image.open(image_upload).convert('RGB')
                image = image.resize((224, 224))  # Model's expected input size
                image.save(image_path)
                
                # Convert to numpy array and normalize
                image_arr = np.array(image).astype(np.float32) / 255.0
                # Add batch dimension
                image_arr = np.expand_dims(image_arr, axis=0)
                
                # Check if input needs quantization (for quantized models)
                input_details = interpreter.get_input_details()
                if input_details[0]['dtype'] == np.uint8:
                    input_scale, input_zero_point = input_details[0]['quantization']
                    image_arr = image_arr / input_scale + input_zero_point
                    image_arr = image_arr.astype(np.uint8)

                app.logger.info(f"Preprocessed image shape: {image_arr.shape}")
                app.logger.info(f"Pixel range: {np.min(image_arr)} - {np.max(image_arr)}")

            except Exception as e:
                app.logger.error(f"Preprocessing failed: {str(e)}")
                raise RuntimeError("Image processing error") from e

            pre_time_ms = (time.time() - start_pre) * 1000

            # --- PREDICTION ---
            start_inf = time.time()
            try:
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], image_arr)
                
                # Run inference
                interpreter.invoke()
                
                # Get output tensor
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                # Handle quantization for output if needed
                if output_details[0]['dtype'] == np.uint8:
                    output_scale, output_zero_point = output_details[0]['quantization']
                    predictions = predictions.astype(np.float32)
                    predictions = (predictions - output_zero_point) * output_scale

                app.logger.info(f"Raw predictions: {predictions}")

                predicted_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_idx]) * 100
                predicted_class = classes[predicted_idx]

                app.logger.info(f"Predicted: {predicted_class} ({confidence:.2f}%)")

            except Exception as e:
                app.logger.error(f"Prediction failed: {str(e)}")
                raise RuntimeError("Model prediction error") from e

            inf_time_ms = (time.time() - start_inf) * 1000
            total_time_ms = (time.time() - start_total) * 1000

            # --- LOGGING ---
            try:
                with open(csv_log_path, mode='a', newline='') as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        name,
                        predicted_class,
                        f"{confidence:.2f}",
                        f"{pre_time_ms:.2f}",
                        f"{inf_time_ms:.2f}",
                        f"{total_time_ms:.2f}"
                    ])
            except Exception as e:
                app.logger.error(f"Logging failed: {str(e)}")

            return render_template(
                'index.html',
                full_filename=full_filename,
                pred_class=predicted_class,
                confidence=f"{confidence:.2f}",
                pre_time=f"{pre_time_ms:.2f}",
                inf_time=f"{inf_time_ms:.2f}",
                total_time=f"{total_time_ms:.2f}"
            )

        except Exception as e:
            app.logger.error(f"Error in request: {str(e)}\n{traceback.format_exc()}")
            flash('Error processing your request. Please try another image.', 'error')
            return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)

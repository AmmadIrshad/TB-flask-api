#Production Setup
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os
import uuid
import logging

# Initialize Flask app
app = Flask(__name__)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize the TB detection model client

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="weUzBKzyD6TYzQL04eBi"
    #api_key=os.getenv("ROBOFLOW_API_KEY")  # Use environment variable for API key
)

# Temporary files directory
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def is_xray_image(file_path):
    try:
        img = Image.open(file_path)
        if img.mode in ['L', 'RGB']:
            return True
    except Exception as e:
        logging.error(f"Image validation error: {e}")
    return False

@app.route('/', methods=['GET'])
def welcome():
    logging.info("Welcome endpoint accessed.")
    return jsonify({"message": "Welcome to the Tuberculosis API! Go to /process-image endpoint for detection."}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']

    if not image_file or image_file.filename == '':
        return jsonify({"error": "Invalid file uploaded"}), 400

    temp_image_path = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.jpg")
    try:
        logging.info(f"Saving file to: {temp_image_path}")
        image_file.save(temp_image_path)
    except Exception as e:
        logging.error(f"File save error: {e}")
        return jsonify({"error": f"Could not save the uploaded image: {str(e)}"}), 500

    if not is_xray_image(temp_image_path):
        os.remove(temp_image_path)
        return jsonify({"error": "Uploaded file is not a valid X-ray image"}), 400

    try:
        result = CLIENT.infer(temp_image_path, model_id="tb-detection-pigkb/1")
        os.remove(temp_image_path)
        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Inference error: {e}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Server error: {error}")
    return jsonify({"error": "An internal server error occurred"}), 500

@app.errorhandler(404)
def not_found_error(error):
    logging.warning(f"404 error: {error}")
    return jsonify({"error": "Resource not found"}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)



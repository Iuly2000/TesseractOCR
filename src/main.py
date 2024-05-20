from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

# Preprocess Images
def preprocess_image(image):
    # Resize image to reduce computation
    image = image.resize((800, 600))

    # Apply enhancement to improve contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)

    # Apply Gaussian blur to reduce noise
    image = image.filter(ImageFilter.GaussianBlur(radius=2))

    return image

# OCR Function
def ocr(image):
    # Perform OCR and get the extracted text
    extracted_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
    return extracted_text

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    # Parse the request JSON data
    data = request.get_json()

    # Get the base64 encoded image data
    image_data_base64 = data.get('image_data')

    # Decode the base64 encoded image data
    image_data = base64.b64decode(image_data_base64)

    # Convert the image data to a BytesIO object
    image_file = BytesIO(image_data)

    # Load the image file
    image = Image.open(image_file)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform OCR on the preprocessed image
    extracted_text = ocr(preprocessed_image)

    # Return the extracted text as a JSON response
    return jsonify({'extracted_text': extracted_text})

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging
import os
from utils import preprocess_uploaded_image, find_similar_places, find_similar_item
from models import load_data, vgg_model

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load preprocessed data
places_data = load_data("PreprocessedImages/Places_DATA_processed_with_features.csv",
                        required_columns=['CityName', 'PlaceName'] + [f'Feature_{i}' for i in range(512)])
items_data = load_data("PreprocessedImages/MOCK_DATAITEMS_processed_with_features.csv",
                       required_columns=['PlaceName', 'ItemName'] + [f'Feature_{i}' for i in range(512)])

# Configure logging
logging.basicConfig(level=logging.DEBUG)
@app.route('/hello', methods=['GET'])
def hello():
    return "Hello, World!", 200

@app.route('/scan_place', methods=['POST'])
def upload_file():
    app.logger.debug(f"Received request: {request.form}, {request.files}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    city_name = request.form.get('city')
    if not city_name:
        return jsonify({'error': 'City name is required'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        user_features = preprocess_uploaded_image(file_path, vgg_model)
        similar_places = find_similar_places(city_name, user_features, places_data)

        logging.debug(f"city_name: {city_name}, similar_places: {similar_places}")

        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'city_name': city_name,
            'similar_places': similar_places
        }), 200

@app.route('/scan_item', methods=['POST'])
def scan_item():
    app.logger.debug(f"Received request: {request.form}, {request.files}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    place_name = request.form.get('place')
    if not place_name:
        return jsonify({'error': 'Place name is required'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        user_features = preprocess_uploaded_image(file_path, vgg_model)
        similar_item = find_similar_item(place_name, user_features, items_data)

        logging.debug(f"place_name: {place_name}, similar_item: {similar_item}")

        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'place_name': place_name,
            'similar_item': similar_item
        }), 200

if __name__ == '__main__':
    app.run(debug=True)

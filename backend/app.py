from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from analyzer import EnhancedCVParserV3
import uuid
import os

app = Flask(__name__, static_folder='frontend')
CORS(app)  # Enable CORS for all domains on all routes

analysis_results = {}  # In-memory storage

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    filename = file.filename

    # Save the file temporarily
    temp_filename = os.path.join('uploads', filename)
    file.save(temp_filename)

    # Analyze the file
    parser = EnhancedCVParserV3()
    analysis = parser.analyze_file(temp_filename)

    # Generate a unique ID for the analysis
    analysis_id = str(uuid.uuid4())
    analysis_results[analysis_id] = analysis

    # Delete the temporary file
    os.remove(temp_filename)

    return jsonify({'analysis_id': analysis_id}), 200

@app.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    analysis = analysis_results.get(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    return jsonify(analysis), 200

@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory('frontend', 'index.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5099)

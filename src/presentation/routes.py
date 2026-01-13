"""
Flask Routes - HTTP request handlers

Organizes routes into blueprints for modularity:
- main_bp: Index page and static files
- upload_bp: File upload handling
- process_bp: Document processing endpoints
- extract_bp: Text extraction endpoints
"""

from flask import Blueprint, render_template, request, jsonify, send_from_directory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..application import DocumentScannerService, TextExtractionService
    from ..infrastructure.file_manager import FileManager


def create_main_blueprint(file_manager: "FileManager") -> Blueprint:
    """
    Create main blueprint for index and sample images.
    
    Routes:
    - GET / : Index page
    - GET /sample_images : List sample images
    - GET /sample_images_serve/<filename> : Serve sample image
    """
    bp = Blueprint('main', __name__)
    
    @bp.route('/')
    def index():
        """Render main index page."""
        return render_template('index.html')
    
    @bp.route('/sample_images')
    def get_sample_images():
        """Get list of available sample images."""
        images = file_manager.list_sample_images()
        return jsonify({'images': images})
    
    @bp.route('/sample_images_serve/<filename>')
    def serve_sample_image(filename):
        """Serve a sample image file."""
        return send_from_directory(file_manager.sample_folder, filename)
    
    return bp


def create_upload_blueprint(file_manager: "FileManager") -> Blueprint:
    """
    Create upload blueprint for file uploads.
    
    Routes:
    - POST /upload : Upload image file
    """
    bp = Blueprint('upload', __name__)
    
    @bp.route('/upload', methods=['POST'])
    def upload_file():
        """
        Handle file upload.
        
        Accepts multipart/form-data with 'file' field.
        Returns JSON with path to uploaded file.
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filepath = file_manager.save_uploaded_file(file)
        return jsonify({'path': filepath})
    
    return bp


def create_process_blueprint(
    opencv_scanner: "DocumentScannerService",
    gemini_scanner: "DocumentScannerService"
) -> Blueprint:
    """
    Create processing blueprint for document scanning.

    Routes:
    - POST /process_opencv : Process with OpenCV
    - POST /process_gemini : Process with Gemini AI
    """
    bp = Blueprint('process', __name__)

    @bp.route('/process_opencv', methods=['POST'])
    def process_opencv():
        """
        Process document using OpenCV corner detection.

        Accepts form data with 'path' field.
        Returns JSON with processing results.
        """
        image_path = request.form.get('path')
        if not image_path:
            return jsonify({'error': 'No path provided'}), 400

        try:
            result = opencv_scanner.scan(image_path)
            return jsonify(result.to_dict())
        except FileNotFoundError:
            return jsonify({'error': 'File not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @bp.route('/process_gemini', methods=['POST'])
    def process_gemini():
        """
        Process document using Gemini AI corner detection.

        Accepts form data with 'path' field.
        Returns JSON with processing results.
        """
        image_path = request.form.get('path')
        if not image_path:
            return jsonify({'error': 'No path provided'}), 400

        try:
            result = gemini_scanner.scan(image_path)
            return jsonify(result.to_dict())
        except FileNotFoundError:
            return jsonify({'error': 'File not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return bp


def create_extract_blueprint(
    text_service: "TextExtractionService",
    api_key_configured: bool
) -> Blueprint:
    """
    Create extraction blueprint for text extraction.

    Routes:
    - POST /extract_text : Extract text from document (Gemini)
    """
    bp = Blueprint('extract', __name__)

    @bp.route('/extract_text', methods=['POST'])
    def extract_text():
        """
        Extract text from document using Gemini OCR.

        Accepts form data with 'path' field.
        Returns JSON with structured text extraction.
        """
        if not api_key_configured:
            return jsonify({'error': 'Gemini API Key not configured'}), 500

        image_path = request.form.get('path')
        if not image_path:
            return jsonify({'error': 'No path provided'}), 400

        result = text_service.extract(image_path)

        if result is None:
            return jsonify({'error': 'Could not extract text'}), 400

        return jsonify(result.to_dict())

    return bp


def create_blueprints(
    file_manager: "FileManager",
    opencv_scanner: "DocumentScannerService",
    gemini_scanner: "DocumentScannerService",
    text_service: "TextExtractionService",
    api_key_configured: bool
) -> list[Blueprint]:
    """
    Create all blueprints with their dependencies.

    Factory function that creates and configures all route blueprints.
    Returns list of blueprints to register with Flask app.
    """
    return [
        create_main_blueprint(file_manager),
        create_upload_blueprint(file_manager),
        create_process_blueprint(opencv_scanner, gemini_scanner),
        create_extract_blueprint(text_service, api_key_configured),
    ]


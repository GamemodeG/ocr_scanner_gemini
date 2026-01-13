"""
OCR Scanner Application - Entry Point

Clean Architecture Flask Application for Document Scanning and OCR.

Architecture:
├── src/
│   ├── domain/          - Business entities and interfaces
│   ├── infrastructure/  - External services (OpenCV, Gemini)
│   ├── application/     - Business logic orchestration
│   └── presentation/    - HTTP routes
│
└── app.py               - This file (composition root)

Author: Selivanov Ivan PI51
"""

from flask import Flask

# Import configuration
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

# Import application components
from src.container import Container
from src.presentation.routes import create_blueprints


def create_app(gemini_api_key: str = None) -> Flask:
    """
    Application factory function.
    
    Creates and configures Flask application with all dependencies.
    This is the composition root where all components are wired together.
    
    Args:
        gemini_api_key: Optional API key override
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Configuration
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['PROCESSED_FOLDER'] = 'static/processed'
    app.config['SAMPLE_FOLDER'] = 'sample_images'
    
    # Use provided key or fallback to config
    api_key = gemini_api_key or GEMINI_API_KEY
    
    # Create dependency injection container
    container = Container(
        gemini_api_key=api_key,
        upload_folder=app.config['UPLOAD_FOLDER'],
        processed_folder=app.config['PROCESSED_FOLDER'],
        sample_folder=app.config['SAMPLE_FOLDER']
    )
    
    # Create blueprints with injected dependencies
    blueprints = create_blueprints(
        file_manager=container.file_manager,
        opencv_scanner=container.opencv_scanner,
        gemini_scanner=container.gemini_scanner if container.has_gemini else container.opencv_scanner,
        text_service=container.text_service if container.has_gemini else None,
        api_key_configured=container.has_gemini
    )
    
    # Register all blueprints
    for blueprint in blueprints:
        app.register_blueprint(blueprint)
    
    return app


# Create application instance
app = create_app()


if __name__ == '__main__':
    print("=" * 60)
    print("OCR Scanner - Clean Architecture")
    print("=" * 60)
    print(f"Gemini API: {'Configured' if GEMINI_API_KEY else 'Not configured'}")
    print("Starting server on http://localhost:8000")
    print("=" * 60)
    
    app.run(debug=True, port=8000)

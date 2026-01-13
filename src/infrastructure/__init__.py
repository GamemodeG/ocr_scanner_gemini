"""
Infrastructure Layer - External Services Implementation

This layer contains concrete implementations of domain interfaces:
- OpenCV-based image processing
- Gemini API client for AI-powered detection and OCR
- File system operations

Infrastructure depends on Domain layer (implements its interfaces)
but Domain never depends on Infrastructure.
"""

from .image_processor import OpenCVImageProcessor
from .corner_detectors import OpenCVCornerDetector, GeminiCornerDetector
from .text_extractors import GeminiTextExtractor
from .file_manager import FileManager

__all__ = [
    "OpenCVImageProcessor",
    "OpenCVCornerDetector",
    "GeminiCornerDetector",
    "GeminiTextExtractor",
    "FileManager",
]


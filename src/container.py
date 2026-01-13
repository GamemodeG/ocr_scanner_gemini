"""
Dependency Injection Container - Wires all components together

This module implements a simple DI container that:
1. Creates all infrastructure components
2. Injects them into application services
3. Provides configured services to presentation layer

This is the composition root - the only place where
concrete implementations are instantiated.
"""

from typing import Optional

from .domain.interfaces import ICornerDetector, IImageProcessor, ITextExtractor
from .infrastructure.image_processor import OpenCVImageProcessor
from .infrastructure.corner_detectors import OpenCVCornerDetector, GeminiCornerDetector
from .infrastructure.text_extractors import GeminiTextExtractor
from .infrastructure.file_manager import FileManager
from .application.scanner_service import DocumentScannerService
from .application.text_extraction_service import TextExtractionService


class Container:
    """
    Dependency Injection Container.
    
    Creates and manages all application components.
    Implements Singleton pattern for shared dependencies.
    """
    
    def __init__(self, 
                 gemini_api_key: Optional[str] = None,
                 upload_folder: str = "static/uploads",
                 processed_folder: str = "static/processed",
                 sample_folder: str = "sample_images"):
        """
        Initialize container with configuration.
        
        Args:
            gemini_api_key: Optional API key for Gemini services
            upload_folder: Directory for uploaded files
            processed_folder: Directory for processed images
            sample_folder: Directory for sample images
        """
        self._gemini_api_key = gemini_api_key
        self._upload_folder = upload_folder
        self._processed_folder = processed_folder
        self._sample_folder = sample_folder
        
        # Lazy-initialized singletons
        self._file_manager: Optional[FileManager] = None
        self._image_processor: Optional[IImageProcessor] = None
        self._opencv_detector: Optional[ICornerDetector] = None
        self._gemini_detector: Optional[ICornerDetector] = None
        self._text_extractor: Optional[ITextExtractor] = None
        self._opencv_scanner: Optional[DocumentScannerService] = None
        self._gemini_scanner: Optional[DocumentScannerService] = None
        self._text_service: Optional[TextExtractionService] = None
    
    @property
    def has_gemini(self) -> bool:
        """Check if Gemini API key is configured."""
        return self._gemini_api_key is not None and len(self._gemini_api_key) > 0
    
    @property
    def file_manager(self) -> FileManager:
        """Get or create FileManager singleton."""
        if self._file_manager is None:
            self._file_manager = FileManager(
                upload_folder=self._upload_folder,
                processed_folder=self._processed_folder,
                sample_folder=self._sample_folder
            )
        return self._file_manager
    
    @property
    def image_processor(self) -> IImageProcessor:
        """Get or create ImageProcessor singleton."""
        if self._image_processor is None:
            self._image_processor = OpenCVImageProcessor()
        return self._image_processor
    
    @property
    def opencv_detector(self) -> ICornerDetector:
        """Get or create OpenCV corner detector singleton."""
        if self._opencv_detector is None:
            self._opencv_detector = OpenCVCornerDetector()
        return self._opencv_detector
    
    @property
    def gemini_detector(self) -> ICornerDetector:
        """Get or create Gemini corner detector singleton."""
        if self._gemini_detector is None:
            if not self.has_gemini:
                raise RuntimeError("Gemini API key not configured")
            self._gemini_detector = GeminiCornerDetector(self._gemini_api_key)
        return self._gemini_detector
    
    @property
    def text_extractor(self) -> ITextExtractor:
        """Get or create text extractor singleton."""
        if self._text_extractor is None:
            if not self.has_gemini:
                raise RuntimeError("Gemini API key not configured for text extraction")
            self._text_extractor = GeminiTextExtractor(self._gemini_api_key)
        return self._text_extractor
    
    @property
    def opencv_scanner(self) -> DocumentScannerService:
        """Get or create OpenCV-based scanner service."""
        if self._opencv_scanner is None:
            self._opencv_scanner = DocumentScannerService(
                image_processor=self.image_processor,
                corner_detector=self.opencv_detector,
                file_manager=self.file_manager
            )
        return self._opencv_scanner
    
    @property
    def gemini_scanner(self) -> DocumentScannerService:
        """Get or create Gemini-based scanner service."""
        if self._gemini_scanner is None:
            self._gemini_scanner = DocumentScannerService(
                image_processor=self.image_processor,
                corner_detector=self.gemini_detector,
                file_manager=self.file_manager
            )
        return self._gemini_scanner
    
    @property
    def text_service(self) -> TextExtractionService:
        """Get or create text extraction service."""
        if self._text_service is None:
            self._text_service = TextExtractionService(
                text_extractor=self.text_extractor,
                file_manager=self.file_manager
            )
        return self._text_service
    


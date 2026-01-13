"""
Text Extraction Service - Handles OCR and text extraction

Coordinates text extraction from document images.
Uses strategy pattern for different extraction backends.
"""

from typing import Optional

from ..domain.entities import TextExtractionResult
from ..domain.interfaces import ITextExtractor
from ..infrastructure.file_manager import FileManager


class TextExtractionService:
    """
    Service for extracting text from document images.
    
    Provides high-level API for text extraction,
    abstracting away the specific extraction implementation.
    """
    
    def __init__(self, 
                 text_extractor: ITextExtractor,
                 file_manager: FileManager):
        """
        Initialize text extraction service.
        
        Args:
            text_extractor: Implementation of text extraction
            file_manager: File system manager
        """
        self._extractor = text_extractor
        self._file_manager = file_manager
    
    def extract(self, image_path: str) -> Optional[TextExtractionResult]:
        """
        Extract text from document image.
        
        Args:
            image_path: Path to document image (can be serve path)
            
        Returns:
            TextExtractionResult with structured text, or None if extraction failed
        """
        # Resolve serve path to actual path
        actual_path = self._file_manager.resolve_serve_path(image_path)
        
        # Check file exists
        if not self._file_manager.file_exists(actual_path):
            return None
        
        # Check cache first
        cached_data = self._file_manager.get_cache(actual_path, suffix="_ocr")
        if cached_data:
            return TextExtractionResult(
                ascii_diagram=cached_data.get('ascii_diagram', ''),
                markdown_text=cached_data.get('markdown_text', ''),
                description=cached_data.get('description', ''),
                seo_keywords=cached_data.get('seo_keywords', '')
            )
        
        # Delegate to extractor
        result = self._extractor.extract(actual_path)
        
        # Save to cache if successful and has data
        if result and (result.ascii_diagram or result.markdown_text or result.description or result.seo_keywords):
            self._file_manager.save_cache(actual_path, result.to_dict(), suffix="_ocr")
            
        return result
    
    @property
    def extractor_name(self) -> str:
        """Get name of the text extractor being used."""
        return self._extractor.name

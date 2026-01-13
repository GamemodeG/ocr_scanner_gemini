"""
Domain Interfaces - Abstract contracts for external services

Interfaces define what the domain layer needs from external services
without specifying how those services are implemented.

This enables:
- Dependency Inversion (depend on abstractions, not concretions)
- Easy testing with mock implementations
- Swappable implementations (OpenCV vs Gemini)
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from .entities import Corners, TextExtractionResult


class ICornerDetector(ABC):
    """
    Interface for document corner detection.
    
    Implementations may use:
    - OpenCV contour detection
    - Gemini Vision API
    - Other ML models
    """
    
    @abstractmethod
    def detect(self, image_path: str) -> Optional[Corners]:
        """
        Detect 4 corners of a document in the image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Corners object with 4 corner coordinates, or None if detection failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the detector."""
        pass


class IImageProcessor(ABC):
    """
    Interface for image processing operations.
    
    Handles low-level image transformations:
    - Grayscale conversion
    - Blurring
    - Edge detection
    - Perspective transform
    """
    
    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        """Load image from path."""
        pass
    
    @abstractmethod
    def save(self, image: np.ndarray, path: str) -> str:
        """Save image to path, return the path."""
        pass
    
    @abstractmethod
    def resize(self, image: np.ndarray, height: int) -> tuple[np.ndarray, float]:
        """
        Resize image to target height, preserving aspect ratio.
        
        Returns:
            Tuple of (resized_image, scale_ratio)
        """
        pass
    
    @abstractmethod
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        pass
    
    @abstractmethod
    def blur(self, image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """Apply Gaussian blur."""
        pass
    
    @abstractmethod
    def detect_edges(self, image: np.ndarray, low: int = 75, high: int = 200) -> np.ndarray:
        """Detect edges using Canny algorithm."""
        pass
    
    @abstractmethod
    def perspective_transform(self, image: np.ndarray, corners: Corners) -> np.ndarray:
        """Apply perspective transform to get top-down view."""
        pass
    
    @abstractmethod
    def binarize(self, image: np.ndarray, block_size: int = 21, c: int = 15) -> np.ndarray:
        """Apply adaptive thresholding for document binarization."""
        pass


class ITextExtractor(ABC):
    """
    Interface for text extraction from documents.
    
    Implementations may use:
    - Tesseract OCR
    - Gemini Vision API
    - Other OCR services
    """
    
    @abstractmethod
    def extract(self, image_path: str) -> Optional[TextExtractionResult]:
        """
        Extract text content from document image.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            TextExtractionResult with structured text, or None if extraction failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the extractor."""
        pass

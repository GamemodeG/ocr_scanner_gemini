"""
Domain Layer - Core Business Entities and Interfaces

This layer contains:
- Entity definitions (Document, Corner, ProcessingResult)
- Abstract interfaces for external services
- Business rules and validation logic

No external dependencies allowed in this layer.
"""

from .entities import Document, Corners, ProcessingStage, ProcessingResult, TextExtractionResult
from .interfaces import ICornerDetector, IImageProcessor, ITextExtractor

__all__ = [
    "Document",
    "Corners", 
    "ProcessingStage",
    "ProcessingResult",
    "TextExtractionResult",
    "ICornerDetector",
    "IImageProcessor",
    "ITextExtractor",
]

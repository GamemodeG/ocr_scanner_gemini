"""
Domain Entities - Core business objects

Entities are the heart of the domain layer.
They represent the core concepts of the application
and contain business rules.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class ProcessingMethod(Enum):
    """Available document processing methods."""
    OPENCV = "opencv"
    GEMINI = "gemini"


@dataclass
class Corners:
    """
    Represents 4 corners of a detected document.
    
    Coordinate system: origin at top-left, x increases right, y increases down.
    Order: top_left -> top_right -> bottom_right -> bottom_left (clockwise)
    """
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]
    
    def to_numpy(self) -> np.ndarray:
        """Convert corners to numpy array of shape (4, 2)."""
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left
        ], dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "Corners":
        """Create Corners from numpy array of shape (4, 2)."""
        if arr.shape != (4, 2):
            raise ValueError(f"Expected shape (4, 2), got {arr.shape}")
        return cls(
            top_left=tuple(arr[0]),
            top_right=tuple(arr[1]),
            bottom_right=tuple(arr[2]),
            bottom_left=tuple(arr[3])
        )
    
    def scale(self, factor: float) -> "Corners":
        """Scale all corners by a factor."""
        return Corners(
            top_left=(self.top_left[0] * factor, self.top_left[1] * factor),
            top_right=(self.top_right[0] * factor, self.top_right[1] * factor),
            bottom_right=(self.bottom_right[0] * factor, self.bottom_right[1] * factor),
            bottom_left=(self.bottom_left[0] * factor, self.bottom_left[1] * factor)
        )


@dataclass
class Document:
    """
    Represents a document image to be processed.
    
    Contains the source path and metadata about the document.
    """
    path: str
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        """Validate document path exists."""
        import os
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Document not found: {self.path}")


@dataclass
class ProcessingStage:
    """
    Represents a single stage in document processing pipeline.
    
    Each stage has a name, description, and resulting image path.
    """
    name: str
    description: str
    image_path: str
    order: int


@dataclass
class ProcessingResult:
    """
    Complete result of document processing.
    
    Contains all processing stages and the final scanned document.
    """
    original_path: str
    contour_path: str
    scanned_path: str
    all_stages_path: str
    stages: dict[str, str] = field(default_factory=dict)
    method: ProcessingMethod = ProcessingMethod.OPENCV
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step1": self.original_path,
            "step2": self.contour_path,
            "step3": self.scanned_path,
            "all_stages": self.all_stages_path,
            "stages": self.stages,
            "method": self.method.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingResult":
        """Create from dictionary."""
        return cls(
            original_path=data.get("step1", ""),
            contour_path=data.get("step2", ""),
            scanned_path=data.get("step3", ""),
            all_stages_path=data.get("all_stages", ""),
            stages=data.get("stages", {}),
            method=ProcessingMethod(data.get("method", "opencv"))
        )


@dataclass
class TextExtractionResult:
    """
    Result of text extraction from document.
    
    Contains structured text in multiple formats.
    """
    ascii_diagram: str = ""
    markdown_text: str = ""
    description: str = ""
    seo_keywords: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ascii_diagram": self.ascii_diagram,
            "markdown_text": self.markdown_text,
            "description": self.description,
            "seo_keywords": self.seo_keywords
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TextExtractionResult":
        """Create from dictionary."""
        return cls(
            ascii_diagram=data.get("ascii_diagram", ""),
            markdown_text=data.get("markdown_text", ""),
            description=data.get("description", ""),
            seo_keywords=data.get("seo_keywords", "")
        )

"""
Application Layer - Use Cases and Business Logic Orchestration

This layer contains:
- Service classes that orchestrate domain logic
- Use case implementations
- Business workflow coordination

Application layer depends on Domain layer.
It uses Infrastructure through Domain interfaces (Dependency Injection).
"""

from .scanner_service import DocumentScannerService
from .text_extraction_service import TextExtractionService

__all__ = [
    "DocumentScannerService",
    "TextExtractionService",
]

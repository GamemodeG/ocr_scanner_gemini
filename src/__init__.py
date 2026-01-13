"""
OCR Scanner - Modular Document Scanner Application

Architecture: Clean Architecture (Hexagonal)
├── domain/          - Business entities and interfaces (no external dependencies)
├── infrastructure/  - External services implementation (OpenCV, Gemini API)
├── application/     - Use cases and business logic orchestration
└── presentation/    - HTTP routes and response formatting

Principles:
- Single Responsibility: each module does one thing
- Dependency Inversion: inner layers don't depend on outer layers
- Separation of Concerns: clear boundaries between layers
"""

__version__ = "2.0.0"
__author__ = "Selivanov Ivan PI51"

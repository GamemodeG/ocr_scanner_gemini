"""
Presentation Layer - HTTP Routes and Response Formatting

This layer contains:
- Flask route handlers (blueprints)
- Request/response serialization
- Error handling and HTTP status codes

Presentation depends on Application layer.
It's the outermost layer - entry point for HTTP requests.
"""

from .routes import create_blueprints

__all__ = [
    "create_blueprints",
]

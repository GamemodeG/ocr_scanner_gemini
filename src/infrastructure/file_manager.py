"""
File Manager - Handles file system operations

Centralizes all file operations:
- Creating directories
- Generating unique filenames
- Resolving paths
"""

import os
from datetime import datetime
from typing import Optional


class FileManager:
    """
    Manages file system operations for the application.
    
    Handles:
    - Upload directory management
    - Processed files directory
    - Sample images directory
    - Unique filename generation
    """
    
    def __init__(self, 
                 upload_folder: str = "static/uploads",
                 processed_folder: str = "static/processed",
                 sample_folder: str = "sample_images"):
        """
        Initialize file manager with folder paths.
        
        Args:
            upload_folder: Directory for uploaded files
            processed_folder: Directory for processed images
            sample_folder: Directory for sample images
        """
        self.upload_folder = upload_folder
        self.processed_folder = processed_folder
        self.sample_folder = sample_folder
        
        # Ensure directories exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)
    
    def generate_filename(self, prefix: str, extension: str = "jpg") -> str:
        """
        Generate unique filename with timestamp.
        
        Args:
            prefix: Prefix for the filename
            extension: File extension (without dot)
            
        Returns:
            Unique filename like "prefix_20231215143022123456.jpg"
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"{prefix}_{timestamp}.{extension}"
    
    def get_upload_path(self, filename: str) -> str:
        """Get full path for upload file."""
        return os.path.join(self.upload_folder, filename)
    
    def get_processed_path(self, filename: str) -> str:
        """Get full path for processed file."""
        return os.path.join(self.processed_folder, filename)
    
    def get_sample_path(self, filename: str) -> str:
        """Get full path for sample file."""
        return os.path.join(self.sample_folder, filename)
    
    def resolve_serve_path(self, serve_path: str) -> str:
        """
        Convert web serve path to actual file system path.
        
        Args:
            serve_path: Path as served to web (e.g., "/sample_images_serve/test.jpg")
            
        Returns:
            Actual file system path
        """
        if serve_path and serve_path.startswith('/sample_images_serve/'):
            filename = serve_path.replace('/sample_images_serve/', '')
            return self.get_sample_path(filename)
        return serve_path
    
    def list_sample_images(self) -> list[dict]:
        """
        List all sample images available.
        
        Returns:
            List of dicts with 'name' and 'path' keys
        """
        if not os.path.exists(self.sample_folder):
            return []
        
        valid_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for filename in os.listdir(self.sample_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_formats:
                images.append({
                    'name': filename,
                    'path': f'/sample_images_serve/{filename}'
                })
        
        return images
    
    def save_uploaded_file(self, file, filename: Optional[str] = None) -> str:
        """
        Save uploaded file to upload directory.
        
        Args:
            file: File object from request
            filename: Optional custom filename
            
        Returns:
            Full path to saved file
        """
        if filename is None:
            filename = self.generate_filename("upload")
        
        filepath = self.get_upload_path(filename)
        file.save(filepath)
        return filepath
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(path)

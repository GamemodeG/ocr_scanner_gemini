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
        import urllib.parse
        
        if not serve_path:
            return serve_path
            
        # Decode URL encoding (e.g. %20 -> space)
        serve_path = urllib.parse.unquote(serve_path)
        
        if serve_path.startswith('/sample_images_serve/'):
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

    def calculate_md5(self, filepath: str) -> str:
        """Calculate MD5 hash of a file."""
        import hashlib
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_cache_path(self, file_hash: str, suffix: str = "_ocr") -> str:
        """Get path to cache file based on hash and suffix."""
        return os.path.join(self.processed_folder, f"{file_hash}{suffix}.json")

    def save_cache(self, filepath: str, data: dict, suffix: str = "_ocr") -> None:
        """Save result to cache file."""
        import json
        file_hash = self.calculate_md5(filepath)
        cache_path = self.get_cache_path(file_hash, suffix)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_cache(self, filepath: str, suffix: str = "_ocr") -> Optional[dict]:
        """Get result from cache if exists."""
        import json
        try:
            file_hash = self.calculate_md5(filepath)
            cache_path = self.get_cache_path(file_hash, suffix)
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Cache read error: {e}")
        return None

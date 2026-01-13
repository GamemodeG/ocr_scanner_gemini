"""
Document Scanner Service - Main business logic orchestrator

Coordinates the document scanning pipeline:
1. Load image
2. Detect document corners
3. Apply perspective transform
4. Apply image enhancements
5. Generate visualization of all stages
"""

from typing import Optional
from datetime import datetime

from ..domain.entities import ProcessingResult, ProcessingMethod, Corners
from ..domain.interfaces import ICornerDetector, IImageProcessor
from ..infrastructure.file_manager import FileManager


class DocumentScannerService:
    """
    Orchestrates the document scanning process.
    
    Uses Strategy pattern for corner detection (OpenCV vs Gemini).
    Follows SRP by delegating actual processing to specialized components.
    """
    
    def __init__(self,
                 image_processor: IImageProcessor,
                 corner_detector: ICornerDetector,
                 file_manager: FileManager):
        """
        Initialize scanner service with dependencies.
        
        Args:
            image_processor: Implementation of image processing operations
            corner_detector: Implementation of corner detection
            file_manager: File system manager
        """
        self._processor = image_processor
        self._detector = corner_detector
        self._file_manager = file_manager
    
    def scan(self, image_path: str) -> ProcessingResult:
        """
        Execute full document scanning pipeline.
        
        Pipeline stages:
        1. Original - input image
        2. Grayscale - color to gray conversion
        3. Blurred - noise reduction
        4. Edges - Canny edge detection
        5. Morphology - edge cleanup
        6. Contour - detected document boundary
        7. Warped - perspective corrected
        8. Sharpened - edge enhancement
        9. Binary - final thresholded result
        
        Args:
            image_path: Path to input image
            
        Returns:
            ProcessingResult with paths to all generated images
        """
        # Resolve path if it's a serve path
        actual_path = self._file_manager.resolve_serve_path(image_path)
        
        # Load original image
        image = self._processor.load(actual_path)
        original = image.copy()
        
        # Resize for processing
        resized, ratio = self._processor.resize(image, height=500)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        
        # Stage storage
        stages = {}
        stage_images = []
        stage_labels = []
        
        # Stage 1: Original
        stages['1_original'] = self._save_stage(resized, timestamp, "1_original")
        stage_images.append(resized.copy())
        stage_labels.append("1.Original")
        
        # Stage 2: Grayscale
        gray = self._processor.to_grayscale(resized)
        stages['2_grayscale'] = self._save_stage(gray, timestamp, "2_grayscale")
        stage_images.append(self._processor.gray_to_bgr(gray))
        stage_labels.append("2.Grayscale")
        
        # Stage 3: Blurred
        blurred = self._processor.blur(gray, kernel_size=7)
        stages['3_blurred'] = self._save_stage(blurred, timestamp, "3_blurred")
        stage_images.append(self._processor.gray_to_bgr(blurred))
        stage_labels.append("3.Blurred")
        
        # Stage 4: Edges
        edges = self._processor.detect_edges(blurred, low=75, high=200)
        stages['4_edges'] = self._save_stage(edges, timestamp, "4_edges")
        stage_images.append(self._processor.gray_to_bgr(edges))
        stage_labels.append("4.Edges")
        
        # Stage 5: Morphology
        morphed = self._processor.morphology_close(edges, kernel_size=5, iterations=2)
        stages['5_morphology'] = self._save_stage(morphed, timestamp, "5_morphology")
        stage_images.append(self._processor.gray_to_bgr(morphed))
        stage_labels.append("5.Morphology")
        
        # Stage 6: Contour detection
        corners = self._detector.detect(actual_path)
        
        if corners is None:
            # Fallback: use entire image
            h, w = resized.shape[:2]
            corners = Corners(
                top_left=(0, 0),
                top_right=(w, 0),
                bottom_right=(w, h),
                bottom_left=(0, h)
            )
        
        # Scale corners to resized image
        scaled_corners = corners.scale(1.0 / ratio)
        
        contour_img = self._processor.draw_contour(resized, scaled_corners)
        
        # Add detector label
        import cv2
        cv2.putText(contour_img, self._detector.name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        stages['6_contour'] = self._save_stage(contour_img, timestamp, "6_contour")
        stage_images.append(contour_img)
        stage_labels.append(f"6.{self._detector.name}")
        
        # Stage 7: Warped (perspective transform)
        warped = self._processor.perspective_transform(original, corners)
        warped_resized, _ = self._processor.resize(warped, height=500)
        stages['7_warped'] = self._save_stage(warped_resized, timestamp, "7_warped")
        stage_images.append(warped_resized)
        stage_labels.append("7.Warped")
        
        # Stage 8: Sharpened
        warped_gray = self._processor.to_grayscale(warped)
        sharpened = self._processor.sharpen(warped_gray)
        sharpened_resized, _ = self._processor.resize(sharpened, height=500)
        stages['8_sharpened'] = self._save_stage(sharpened_resized, timestamp, "8_sharpened")
        stage_images.append(self._processor.gray_to_bgr(sharpened_resized))
        stage_labels.append("8.Sharpened")
        
        # Stage 9: Binary
        binary = self._processor.binarize(sharpened, block_size=21, c=15)
        binary_resized, _ = self._processor.resize(binary, height=500)
        stages['9_binary'] = self._save_stage(binary_resized, timestamp, "9_binary")
        stage_images.append(self._processor.gray_to_bgr(binary_resized))
        stage_labels.append("9.Binary")
        
        # Final scanned document (full resolution)
        stages['scanned'] = self._save_stage(warped, timestamp, "scanned")
        
        # Create grid visualization
        grid = self._processor.create_grid(stage_images, stage_labels, cell_size=(200, 200), cols=3)
        stages['all_stages'] = self._save_stage(grid, timestamp, "ALL_STAGES")
        
        # Determine method used
        method = ProcessingMethod.GEMINI if "Gemini" in self._detector.name else ProcessingMethod.OPENCV
        
        return ProcessingResult(
            original_path=stages['1_original'],
            contour_path=stages['6_contour'],
            scanned_path=stages['scanned'],
            all_stages_path=stages['all_stages'],
            stages={
                'original': stages['1_original'],
                'grayscale': stages['2_grayscale'],
                'blurred': stages['3_blurred'],
                'edges': stages['4_edges'],
                'morphology': stages['5_morphology'],
                'contour': stages['6_contour'],
                'warped': stages['7_warped'],
                'sharpened': stages['8_sharpened'],
                'binary': stages['9_binary'],
                'scanned': stages['scanned'],
                'all_stages': stages['all_stages']
            },
            method=method
        )
    
    def _save_stage(self, image, timestamp: str, stage_name: str) -> str:
        """Save processing stage image and return path."""
        filename = f"{timestamp}_{stage_name}.jpg"
        path = self._file_manager.get_processed_path(filename)
        return self._processor.save(image, path)

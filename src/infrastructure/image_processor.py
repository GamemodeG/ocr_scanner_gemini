"""
OpenCV Image Processor - Implementation of IImageProcessor

Handles all low-level image operations using OpenCV library.
Each method does exactly one thing (SRP).
"""

import cv2
import numpy as np
from typing import Optional

from ..domain.interfaces import IImageProcessor
from ..domain.entities import Corners


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points in clockwise order starting from top-left.
    
    Algorithm:
    - Top-left has smallest sum (x + y)
    - Bottom-right has largest sum
    - Top-right has smallest difference (y - x)
    - Bottom-left has largest difference
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


class OpenCVImageProcessor(IImageProcessor):
    """
    OpenCV-based image processor.
    
    Implements all basic image operations needed for document scanning:
    - Loading/saving images
    - Color space conversions
    - Filtering and edge detection
    - Geometric transformations
    """
    
    def load(self, path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Returns BGR image (OpenCV default format).
        Raises FileNotFoundError if image cannot be loaded.
        """
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return image
    
    def save(self, image: np.ndarray, path: str) -> str:
        """
        Save image to file path.
        
        Supports: jpg, png, bmp formats (determined by extension).
        Returns the path where image was saved.
        """
        cv2.imwrite(path, image)
        return path
    
    def resize(self, image: np.ndarray, height: int) -> tuple[np.ndarray, float]:
        """
        Resize image to target height, preserving aspect ratio.
        
        Args:
            image: Input image
            height: Target height in pixels
            
        Returns:
            Tuple of (resized_image, scale_ratio)
            scale_ratio = original_height / new_height
        """
        h, w = image.shape[:2]
        ratio = h / float(height)
        new_width = int(w / ratio)
        resized = cv2.resize(image, (new_width, height))
        return resized, ratio
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to grayscale."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def blur(self, image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image (grayscale or color)
            kernel_size: Size of Gaussian kernel (must be odd)
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def detect_edges(self, image: np.ndarray, low: int = 75, high: int = 200) -> np.ndarray:
        """
        Detect edges using Canny algorithm.
        
        Args:
            image: Input image (should be grayscale and blurred)
            low: Lower threshold for hysteresis
            high: Upper threshold for hysteresis
        """
        return cv2.Canny(image, low, high)
    
    def morphology_close(self, image: np.ndarray, kernel_size: int = 5, iterations: int = 2) -> np.ndarray:
        """
        Apply morphological closing to fill gaps in edges.
        
        Dilation followed by erosion helps connect broken contours.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(image, kernel, iterations=iterations)
        eroded = cv2.erode(dilated, kernel, iterations=iterations - 1)
        return eroded
    
    def perspective_transform(self, image: np.ndarray, corners: Corners) -> np.ndarray:
        """
        Apply perspective transform to get top-down (bird's eye) view.
        
        Transforms the quadrilateral defined by corners into a rectangle.
        """
        pts = corners.to_numpy()
        rect = _order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Calculate output dimensions
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Destination points for transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        # Compute and apply transform
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        
        Enhances edges by subtracting a blurred version from the original.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return sharpened
    
    def binarize(self, image: np.ndarray, block_size: int = 21, c: int = 15) -> np.ndarray:
        """
        Apply adaptive thresholding for document binarization.
        
        Creates a black/white image optimal for OCR.
        
        Args:
            image: Grayscale input image
            block_size: Size of pixel neighborhood for threshold calculation
            c: Constant subtracted from mean
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
        
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c
        )
    
    def draw_contour(self, image: np.ndarray, corners: Corners, color: tuple = (0, 255, 0), thickness: int = 3) -> np.ndarray:
        """
        Draw document contour on image.
        
        Draws the quadrilateral and marks corner points.
        """
        result = image.copy()
        pts = corners.to_numpy().astype(int)
        
        # Draw polygon
        cv2.polylines(result, [pts], True, color, thickness)
        
        # Draw corner points
        for point in pts:
            cv2.circle(result, tuple(point), 8, (0, 0, 255), -1)
        
        return result
    
    def gray_to_bgr(self, image: np.ndarray) -> np.ndarray:
        """Convert grayscale image to BGR for visualization."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    
    def create_grid(self, images: list[np.ndarray], labels: list[str], 
                    cell_size: tuple[int, int] = (200, 200), cols: int = 3) -> np.ndarray:
        """
        Create a grid visualization of multiple images.
        
        Args:
            images: List of images to display
            labels: Labels for each image
            cell_size: (width, height) of each cell
            cols: Number of columns in grid
        """
        cell_w, cell_h = cell_size
        rows = (len(images) + cols - 1) // cols
        
        # Create canvas
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            row, col = idx // cols, idx % cols
            
            # Convert to BGR if needed
            img_bgr = self.gray_to_bgr(img)
            
            # Resize to fit cell
            h, w = img_bgr.shape[:2]
            scale = min(cell_w / w, cell_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img_bgr, (new_w, new_h))
            
            # Center in cell
            x_offset = col * cell_w + (cell_w - new_w) // 2
            y_offset = row * cell_h + (cell_h - new_h) // 2
            
            grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            
            # Add label
            text_x = col * cell_w + 5
            text_y = row * cell_h + 20
            cv2.putText(grid, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return grid

"""
Corner Detectors - Implementations of ICornerDetector

Two implementations:
1. OpenCVCornerDetector - Traditional computer vision approach
2. GeminiCornerDetector - AI-powered using Gemini Vision API
"""

import cv2
import numpy as np
from typing import Optional
import json
import io

from ..domain.interfaces import ICornerDetector
from ..domain.entities import Corners


class OpenCVCornerDetector(ICornerDetector):
    """
    OpenCV-based document corner detection.
    
    Uses edge detection and contour finding to locate document boundaries.
    Tries multiple strategies for robust detection.
    """
    
    @property
    def name(self) -> str:
        return "OpenCV"
    
    def detect(self, image_path: str) -> Optional[Corners]:
        """
        Detect document corners using OpenCV contour detection.
        
        Strategy:
        1. Convert to grayscale
        2. Apply blur and edge detection
        3. Find contours
        4. Approximate to quadrilateral
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Try multiple detection strategies
        corners = self._try_canny_detection(image)
        if corners is not None:
            return corners
        
        corners = self._try_adaptive_threshold(image)
        if corners is not None:
            return corners
        
        corners = self._try_morphological_gradient(image)
        return corners
    
    def _try_canny_detection(self, image: np.ndarray) -> Optional[Corners]:
        """Try edge detection with various Canny parameters."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        canny_params = [(30, 100), (50, 150), (75, 200), (20, 80)]
        blur_sizes = [(5, 5), (7, 7), (3, 3)]
        
        for blur_size in blur_sizes:
            blurred = cv2.GaussianBlur(gray, blur_size, 0)
            
            for low, high in canny_params:
                edges = cv2.Canny(blurred, low, high)
                
                # Morphological closing
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                edges = cv2.dilate(edges, kernel, iterations=2)
                edges = cv2.erode(edges, kernel, iterations=1)
                
                corners = self._find_quad_contour(edges, image.shape)
                if corners is not None:
                    return corners
        
        return None
    
    def _try_adaptive_threshold(self, image: np.ndarray) -> Optional[Corners]:
        """Try adaptive thresholding for edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        for block_size in [11, 21, 31]:
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, 2
            )
            thresh = cv2.bitwise_not(thresh)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            corners = self._find_quad_contour(thresh, image.shape)
            if corners is not None:
                return corners
        
        return None
    
    def _try_morphological_gradient(self, image: np.ndarray) -> Optional[Corners]:
        """Try morphological gradient for edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return self._find_quad_contour(thresh, image.shape)
    
    def _find_quad_contour(self, edges: np.ndarray, image_shape: tuple) -> Optional[Corners]:
        """Find quadrilateral contour in edge image."""
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area, keep top 10
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        h, w = image_shape[:2]
        min_area = h * w * 0.1  # At least 10% of image
        
        epsilons = [0.02, 0.03, 0.04, 0.05, 0.01]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            
            for eps in epsilons:
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2).astype(np.float32)
                    return self._order_corners(pts)
            
            # Try convex hull
            hull = cv2.convexHull(contour)
            for eps in epsilons:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    return self._order_corners(pts)
        
        return None
    
    def _order_corners(self, pts: np.ndarray) -> Corners:
        """Order points as top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return Corners.from_numpy(rect)


class GeminiCornerDetector(ICornerDetector):
    """
    Gemini Vision API-based document corner detection.
    
    Использует генерацию изображения с зелёными точками на углах документа,
    затем извлекает координаты точек с помощью OpenCV цветовой фильтрации.
    
    Алгоритм:
    1. Отправляем фото в Gemini 3 Pro Image с промптом нарисовать зелёные точки
    2. Получаем сгенерированное изображение с точками
    3. Находим зелёные точки через HSV фильтрацию
    4. Проецируем координаты на оригинальное изображение
    """
    
    def __init__(self, api_key: str):
        """
        Инициализация Gemini детектора.
        
        Args:
            api_key: Google Gemini API key
        """
        self._api_key = api_key
        self._client = None
    
    @property
    def name(self) -> str:
        return "Gemini + Nanobanana Pro"
    
    def _get_client(self):
        """Ленивая инициализация Gemini клиента."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client
    
    def detect(self, image_path: str) -> Optional[Corners]:
        """
        Детектирует углы документа через генерацию изображения с точками.
        
        Шаги:
        1. Генерируем изображение с зелёными точками на углах
        2. Находим зелёные точки на сгенерированном изображении
        3. Масштабируем координаты к размеру оригинала
        """
        from google.genai import types
        from PIL import Image
        import os
        
        try:
            client = self._get_client()
            
            # Загружаем оригинальное изображение
            original_image = Image.open(image_path)
            orig_width, orig_height = original_image.size
            
            # Конвертируем в bytes для отправки
            buffered = io.BytesIO()
            original_image.save(buffered, format="JPEG", quality=95)
            image_bytes = buffered.getvalue()
            
            # Промпт для генерации изображения с точками
            prompt = """Найди углы на этом фото для скана, тебе нужно как бы определить, где лист, смотри внимательно.
Найди РОВНО 4 угла (иногда углы могут уходить за края, поэтому рисуй их у края).
Нарисуй яркую ЗЕЛЁНУЮ точку (#00FF00) на КАЖДОМ из 4 углов.
Нарисуй КРАСНУЮ линию (#FF0000), соединяющую все 4 угла (образуя прямоугольник/четырёхугольник)."""
            
            # Создаём Part для изображения
            image_part = types.Part.from_bytes(
                data=image_bytes, 
                mime_type="image/jpeg"
            )
            
            # Генерируем изображение с точками через Gemini 3 Pro Image
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                ),
            )
            
            # Извлекаем сгенерированное изображение
            generated_image = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img_data = part.inline_data.data
                    generated_image = Image.open(io.BytesIO(img_data))
                    break
            
            if generated_image is None:
                print("Gemini не вернул изображение")
                return None
            
            # Сохраняем сгенерированное изображение для отладки
            debug_dir = "static/processed"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "DEBUG_gemini_corners.jpg")
            generated_image.save(debug_path, quality=95)
            print(f"DEBUG: Сохранено изображение с точками -> {debug_path}")
            
            # Получаем размеры сгенерированного изображения
            gen_width, gen_height = generated_image.size
            
            # Находим зелёные точки на сгенерированном изображении
            green_points = self._find_green_points(generated_image)
            
            print(f"DEBUG: Найдено {len(green_points)} зелёных точек")
            
            if len(green_points) != 4:
                print(f"Найдено {len(green_points)} точек вместо 4")
                # Если точек не 4, пробуем взять 4 самые яркие
                if len(green_points) > 4:
                    green_points = green_points[:4]
                elif len(green_points) < 4:
                    return None
            
            # Масштабируем координаты к размеру оригинала
            scale_x = orig_width / gen_width
            scale_y = orig_height / gen_height
            
            scaled_points = []
            for (x, y) in green_points:
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                scaled_points.append((scaled_x, scaled_y))
            
            # Упорядочиваем точки: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(np.array(scaled_points, dtype=np.float32))
            
            return corners
            
        except Exception as e:
            print(f"Gemini corner detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_green_points(self, image: "Image.Image") -> list[tuple[int, int]]:
        """
        Находит зелёные точки на изображении.
        Простой алгоритм: HSV фильтрация по зелёному цвету.
        """
        import os
        
        # Конвертируем PIL в OpenCV формат (BGR)
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Конвертируем в HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Узкий диапазон для ярко-зелёного (#00FF00)
        # H=60° в HSV (в OpenCV = 60/2 = 30), берём узкий диапазон 30-90
        lower_green = np.array([35, 150, 150])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Сохраняем маску для отладки
        debug_mask_path = "static/processed/DEBUG_green_mask.jpg"
        cv2.imwrite(debug_mask_path, mask)
        print(f"DEBUG: Сохранена маска зелёного -> {debug_mask_path}")
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"DEBUG: Найдено {len(contours)} зелёных контуров")
        
        # Извлекаем центры ВСЕХ контуров (без фильтрации по размеру)
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                points.append((cx, cy, area))
                print(f"DEBUG: Точка ({cx}, {cy}) площадь={area}")
        
        # Сортируем по площади и берём 4 самых крупных
        points.sort(key=lambda p: p[2], reverse=True)
        result = [(p[0], p[1]) for p in points[:4]]
        
        print(f"DEBUG: Финальные 4 точки: {result}")
        return result
    
    def _order_corners(self, pts: np.ndarray) -> Corners:
        """
        Упорядочивает точки: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: numpy array формы (4, 2) с координатами точек
            
        Returns:
            Corners объект с упорядоченными углами
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Сумма координат: минимум = top-left, максимум = bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Разность координат: минимум = top-right, максимум = bottom-left
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return Corners.from_numpy(rect)


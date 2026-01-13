#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задание 04: Сканирование и предобработка текста (OCR)

Автор: Селиванов Иван ПИ51

Простой сканер документов с использованием OpenCV:
- Автоматическое определение 4 углов листа бумаги
- Перспективное преобразование (Perspective Transform)
- Бинаризация для четкости текста

Использование:
    python document_scanner.py --image path/to/image.jpg
    python document_scanner.py --image path/to/image.jpg --debug
"""

import cv2
import numpy as np
import argparse
import os

# Опциональный импорт Gemini детектора
GEMINI_AVAILABLE = False
try:
    from gemini_detector import detect_corners_with_gemini
    from config import GEMINI_API_KEY
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
        GEMINI_AVAILABLE = True
except ImportError:
    pass


def order_points(pts):
    """
    Упорядочивает 4 точки в порядке: 
    верхний-левый, верхний-правый, нижний-правый, нижний-левый
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Сумма координат: минимальная = верхний-левый, максимальная = нижний-правый
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Разность координат: минимальная = верхний-правый, максимальная = нижний-левый
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def four_point_transform(image, pts):
    """
    Выполняет перспективное преобразование изображения по 4 точкам
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Вычисляем ширину нового изображения
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Вычисляем высоту нового изображения
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Целевые точки для преобразования (вид сверху)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Матрица перспективного преобразования
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def find_document_contour(image):
    """
    Находит контур документа (4 угла листа бумаги) на изображении.
    Использует несколько стратегий для более надёжной детекции.
    """
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Сохраняем edges для отладки (будет обновляться)
    best_edges = None
    document_contour = None
    
    # Стратегия 1: Различные параметры Canny с Gaussian blur
    canny_params = [
        (30, 100),   # Низкий порог - более чувствительный
        (50, 150),   # Средний порог
        (75, 200),   # Высокий порог
        (20, 80),    # Очень низкий порог
    ]
    
    blur_sizes = [(5, 5), (7, 7), (3, 3)]
    
    for blur_size in blur_sizes:
        blurred = cv2.GaussianBlur(gray, blur_size, 0)
        
        for low, high in canny_params:
            edged = cv2.Canny(blurred, low, high)
            
            # Морфологические операции
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.dilate(edged, kernel, iterations=2)
            edged = cv2.erode(edged, kernel, iterations=1)
            
            if best_edges is None:
                best_edges = edged
            
            contour = _find_quad_contour(edged, image.shape)
            if contour is not None:
                return contour, edged
    
    # Стратегия 2: Bilateral filter (сохраняет края лучше)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    for low, high in canny_params:
        edged = cv2.Canny(bilateral, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.dilate(edged, kernel, iterations=2)
        edged = cv2.erode(edged, kernel, iterations=1)
        
        contour = _find_quad_contour(edged, image.shape)
        if contour is not None:
            return contour, edged
    
    # Стратегия 3: Адаптивный порог вместо Canny
    for block_size in [11, 21, 31]:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, 2
        )
        thresh = cv2.bitwise_not(thresh)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        contour = _find_quad_contour(thresh, image.shape)
        if contour is not None:
            return contour, thresh
    
    # Стратегия 4: Морфологический градиент
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contour = _find_quad_contour(thresh, image.shape)
    if contour is not None:
        return contour, thresh
    
    return None, best_edges


def _find_quad_contour(edged, image_shape):
    """
    Ищет четырёхугольный контур в бинарном изображении
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Сортируем по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    h, w = image_shape[:2]
    min_area = h * w * 0.1  # Минимум 10% изображения
    
    # Пробуем разные epsilon для аппроксимации
    epsilons = [0.02, 0.03, 0.04, 0.05, 0.01]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        
        for eps in epsilons:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            
            if len(approx) == 4:
                # Проверяем, что это выпуклый четырёхугольник
                if cv2.isContourConvex(approx):
                    return approx
        
        # Если не нашли 4-угольник, попробуем convex hull
        hull = cv2.convexHull(contour)
        for eps in epsilons:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx
    
    return None


def create_visualization_grid(images_dict, grid_size=(3, 3), cell_size=(400, 300)):
    """
    Создаёт сетку визуализации из словаря изображений с подписями
    
    Args:
        images_dict: dict с ключами-названиями и значениями-изображениями
        grid_size: (cols, rows) размер сетки
        cell_size: (width, height) размер каждой ячейки
    """
    cols, rows = grid_size
    cell_w, cell_h = cell_size
    
    # Создаём пустой холст
    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
    
    items = list(images_dict.items())
    
    for idx, (title, img) in enumerate(items):
        if idx >= cols * rows:
            break
        
        row = idx // cols
        col = idx % cols
        
        # Конвертируем в BGR если нужно
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        
        # Масштабируем изображение под ячейку (с отступом для текста)
        text_margin = 30
        target_h = cell_h - text_margin - 10
        target_w = cell_w - 10
        
        h, w = img_color.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img_color, (new_w, new_h))
        
        # Позиция в сетке
        x_offset = col * cell_w + (cell_w - new_w) // 2
        y_offset = row * cell_h + text_margin + (target_h - new_h) // 2
        
        # Вставляем изображение
        grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Добавляем подпись
        text_x = col * cell_w + 5
        text_y = row * cell_h + 22
        cv2.putText(grid, f"{idx + 1}. {title}", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return grid


def scan_document(image_path, output_dir="output", debug=False, use_gemini=False):
    """
    Основная функция сканирования документа с полной визуализацией всех этапов
    
    Args:
        image_path: путь к изображению
        output_dir: директория для сохранения результатов
        debug: если True, сохраняет промежуточные этапы и визуализацию
        use_gemini: если True, использует Gemini Vision API для определения контуров
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return False
    
    orig = image.copy()
    
    # Создаём директорию для вывода
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Словарь для хранения этапов визуализации
    stages = {}
    
    # === ЭТАП 1: Оригинал ===
    stages["Оригинал"] = orig.copy()
    
    # Масштабируем для обработки
    height = 500.0
    ratio = image.shape[0] / height
    resized = cv2.resize(image, (int(image.shape[1] / ratio), int(height)))
    
    # === ЭТАП 2: Оттенки серого ===
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    stages["Grayscale"] = gray.copy()
    
    # === ЭТАП 3: Размытие ===
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    stages["Gaussian Blur"] = blurred.copy()
    
    # === ЭТАП 4: Детекция границ ===
    edges_raw = cv2.Canny(blurred, 50, 150)
    stages["Canny Edges"] = edges_raw.copy()
    
    # === ЭТАП 5: Морфологические операции ===
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_morph = cv2.dilate(edges_raw, kernel, iterations=2)
    edges_morph = cv2.erode(edges_morph, kernel, iterations=1)
    stages["Morphology"] = edges_morph.copy()
    
    # Находим контур документа
    contour = None
    edges = edges_morph
    
    # Пробуем Gemini если включён
    if use_gemini and GEMINI_AVAILABLE:
        print("  → Определение контуров через Gemini Vision API...")
        gemini_corners = detect_corners_with_gemini(image_path, GEMINI_API_KEY)
        if gemini_corners is not None:
            # Масштабируем координаты к размеру resized
            scale_x = resized.shape[1] / orig.shape[1]
            scale_y = resized.shape[0] / orig.shape[0]
            contour = gemini_corners.copy()
            contour[:, 0] *= scale_x
            contour[:, 1] *= scale_y
            stages["Gemini Detection"] = stages.get("Canny Edges", edges_raw)
    
    # Fallback на OpenCV если Gemini не сработал
    if contour is None:
        if use_gemini:
            print("  → Gemini не нашёл контур, используем OpenCV...")
        contour, edges = find_document_contour(resized)
    
    if contour is None:
        print(f"Предупреждение: контур документа не найден в {image_path}")
        print("Используем всё изображение...")
        h, w = resized.shape[:2]
        contour = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    
    # === ЭТАП 6: Найденный контур ===
    contour_image = resized.copy()
    cv2.drawContours(contour_image, [contour.astype(int)], -1, (0, 255, 0), 3)
    for point in contour.reshape(-1, 2):
        cv2.circle(contour_image, tuple(point.astype(int)), 8, (0, 0, 255), -1)
    stages["Contour Detection"] = contour_image
    
    # Масштабируем контур обратно к оригинальному размеру
    contour_scaled = contour.reshape(4, 2) * ratio
    
    # === ЭТАП 7: Перспективное преобразование ===
    warped = four_point_transform(orig, contour_scaled)
    stages["Perspective Transform"] = warped.copy()
    
    # === ЭТАП 8: Повышение резкости ===
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.GaussianBlur(gray_warped, (0, 0), 3)
    sharpened = cv2.addWeighted(gray_warped, 1.5, sharpened, -0.5, 0)
    stages["Sharpening"] = sharpened.copy()
    
    # === ЭТАП 9: Бинаризация ===
    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 15
    )
    stages["Binarization"] = binary.copy()
    
    # Сохраняем результат
    output_path = f"{output_dir}/{basename}_scanned.jpg"
    cv2.imwrite(output_path, binary)
    
    if debug:
        # Сохраняем отдельные этапы
        cv2.imwrite(f"{output_dir}/{basename}_1_original.jpg", orig)
        cv2.imwrite(f"{output_dir}/{basename}_2_grayscale.jpg", gray)
        cv2.imwrite(f"{output_dir}/{basename}_3_blurred.jpg", blurred)
        cv2.imwrite(f"{output_dir}/{basename}_4_edges.jpg", edges_raw)
        cv2.imwrite(f"{output_dir}/{basename}_5_morphology.jpg", edges_morph)
        cv2.imwrite(f"{output_dir}/{basename}_6_contour.jpg", contour_image)
        cv2.imwrite(f"{output_dir}/{basename}_7_warped.jpg", warped)
        cv2.imwrite(f"{output_dir}/{basename}_8_sharpened.jpg", sharpened)
        cv2.imwrite(f"{output_dir}/{basename}_9_binary.jpg", binary)
        
        # Создаём визуализацию-сетку всех этапов
        grid = create_visualization_grid(stages, grid_size=(3, 3), cell_size=(450, 350))
        grid_path = f"{output_dir}/{basename}_ALL_STAGES.jpg"
        cv2.imwrite(grid_path, grid)
        print(f"  → Визуализация всех этапов: {grid_path}")
    
    print(f"✓ Обработано: {image_path}")
    print(f"  → Сохранено: {output_path}")
    
    if debug:
        print(f"  → Debug файлы (9 этапов) сохранены в {output_dir}/")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Сканер документов с перспективным преобразованием и бинаризацией"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Путь к изображению для сканирования"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Директория для сохранения результатов (по умолчанию: output)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Сохранять промежуточные этапы обработки"
    )
    parser.add_argument(
        "--gemini", "-g",
        action="store_true",
        help="Использовать Gemini Vision API для определения контуров (требует config.py с API ключом)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Ошибка: файл не найден: {args.image}")
        return 1
    
    # Проверяем доступность Gemini
    use_gemini = False
    if args.gemini:
        if GEMINI_AVAILABLE:
            use_gemini = True
            print("Используется Gemini Vision API для определения контуров")
        else:
            print("Предупреждение: Gemini недоступен. Проверьте config.py с GEMINI_API_KEY")
            print("Используется OpenCV...")
    
    success = scan_document(args.image, args.output, args.debug, use_gemini)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

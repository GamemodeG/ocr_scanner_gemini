#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Scanner - Alternative implementation with LSD (Line Segment Detector)
文档扫描器 - 使用 LSD（线段检测器）的替代实现
Сканер документов - Альтернативная реализация с LSD (детектор линейных сегментов)

Author / 作者 / Автор: Selivanov Ivan PI51 / Селиванов Иван ПИ51

USAGE / 用法 / Использование:
    python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
    
Examples / 示例 / Примеры:
    # Scan single image with interactive mode / 交互模式扫描单张图片 / Сканирование одного изображения с интерактивным режимом:
    python scan.py --image sample_images/desk.JPG -i
    
    # Scan all images in directory / 扫描目录中的所有图片 / Сканирование всех изображений в директории:
    python scan.py --images sample_images

Output / 输出 / Вывод:
    Scanned images will be saved to 'output' directory
    扫描后的图像将保存到 'output' 目录
    Отсканированные изображения сохраняются в директорию 'output'
"""

from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import polygon_interacter as poly_i
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd

import argparse
import os

class DocScanner(object):
    """
    Document scanner class using Line Segment Detection (LSD)
    使用线段检测（LSD）的文档扫描器类
    Класс сканера документов с использованием детекции линейных сегментов (LSD)
    """

    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        Initialize document scanner / 初始化文档扫描器 / Инициализация сканера документов
        
        Args / 参数 / Параметры:
            interactive (bool): If True, user can adjust contour in matplotlib window
                如果为True，用户可以在matplotlib窗口中调整轮廓
                Если True, пользователь может корректировать контур в окне matplotlib
            MIN_QUAD_AREA_RATIO (float): Min area ratio for valid contour (default: 0.25)
                有效轮廓的最小面积比（默认值：0.25）
                Минимальное соотношение площади для действительного контура (по умолчанию: 0.25)
            MAX_QUAD_ANGLE_RANGE (int): Max angle range for valid quadrilateral (default: 40)
                有效四边形的最大角度范围（默认值：40）
                Максимальный диапазон углов для действительного четырёхугольника (по умолчанию: 40)
        """
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE        

    def filter_corners(self, corners, min_dist=20):
        """
        Filters corners that are within min_dist of each other
        过滤距离小于min_dist的角点
        Фильтрует углы, расположенные ближе min_dist друг к другу
        """
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        """
        Returns angle between two vectors in degrees
        返回两个向量之间的角度（度数）
        Возвращает угол между двумя векторами в градусах
        """
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """
        Returns the angle between line segments p2->p1 and p2->p3 in degrees
        返回线段 p2->p1 和 p2->p3 之间的角度（度数）
        Возвращает угол между отрезками p2->p1 и p2->p3 в градусах
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        """
        Returns the range between max and min interior angles of quadrilateral.
        返回四边形内角的最大值和最小值之差。
        Возвращает разницу между максимальным и минимальным внутренними углами четырёхугольника.
        
        Args / 参数 / Аргументы:
            quad: numpy array with vertices ordered clockwise from top-left
                  从左上角顺时针排列顶点的numpy数组
                  numpy массив с вершинами по часовой стрелке от верхнего левого угла
        """
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)          

    def get_corners(self, img):
        """
        Returns a list of corners ((x, y) tuples) found in the input image.
        Uses Line Segment Detection (LSD) algorithm for robust corner detection.
        
        返回在输入图像中找到的角点列表（(x, y) 元组）。
        使用线段检测（LSD）算法进行稳健的角点检测。
        
        Возвращает список углов ((x, y) кортежей), найденных на изображении.
        Использует алгоритм LSD для надёжного обнаружения углов.
        
        Algorithm / 算法 / Алгоритм:
        1. Separate lines into horizontal/vertical / 将线条分为水平/垂直 / Разделить линии на горизонтальные/вертикальные
        2. Draw lines on separate canvases / 在单独的画布上绘制线条 / Нарисовать линии на отдельных холстах
        3. Find connected components / 查找连通组件 / Найти связанные компоненты
        4. Get bounding boxes as final lines / 获取边界框作为最终线条 / Получить ограничивающие рамки как итоговые линии
        5. Line endpoints are corners / 线条端点是角点 / Концы линий являются углами
        6. Line intersections are also corners / 线条交点也是角点 / Пересечения линий также являются углами
        """
        lines = lsd(img)

        # Process LSD output / 处理LSD输出 / Обработка вывода LSD
        # LSD works on edges - combine edges back into lines
        # LSD处理边缘 - 将边缘合并回线条
        # LSD работает с гранями - объединяем грани обратно в линии

        corners = []
        if lines is not None:
            # Separate horizontal and vertical lines / 分离水平和垂直线 / Разделяем горизонтальные и вертикальные линии
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (int(max(x1 - 5, 0)), int(y1)), (int(min(x2 + 5, img.shape[1] - 1)), int(y2)), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (int(x1), int(max(y1 - 5, 0))), (int(x2), int(min(y2 + 5, img.shape[0] - 1))), 255, 2)

            lines = []

            # Find horizontal lines / 查找水平线 / Находим горизонтальные линии
            # connected-components -> bounding boxes -> final lines
            # 连通组件 -> 边界框 -> 最终线条
            # связанные компоненты -> ограничивающие рамки -> итоговые линии
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # Find vertical lines / 查找垂直线 / Находим вертикальные линии
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # Find corner intersections / 查找角点交叉点 / Находим пересечения углов
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # Remove corners in close proximity / 删除过于接近的角点 / Удаляем слишком близкие углы
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """
        Returns True if contour meets all validation requirements
        如果轮廓满足所有验证要求，则返囮True
        Возвращает True, если контур соответствует всем требованиям
        """
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)


    def get_contour(self, rescaled_image):
        """
        Returns a numpy array of shape (4, 2) containing the vertices of the four corners
        of the document in the image. It considers the corners returned from get_corners()
        and uses heuristics to choose the four corners that most likely represent
        the corners of the document. If no corners were found, or the four corners represent
        a quadrilateral that is too small or convex, it returns the original four corners.
        """

        # these constants are carefully chosen
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)

            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # for debugging: uncomment the code below to draw the corners and countour found 
            # by get_corners() and overlay it on the image

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # also attempt to find contours directly from the edged image, which occasionally 
        # produces better results
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Drag the corners of the box to the corners of the document. \n'
            'Close the window when finished.'))
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path):

        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(image_path)

        assert(image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(RESCALED_HEIGHT))

        # get the contour of the document
        screenCnt = self.get_contour(rescaled_image)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # apply the perspective transformation
        warped = transform.four_point_transform(orig, screenCnt * ratio)

        # convert the warped image to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # sharpen image
        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # apply adaptive threshold to get black and white effect
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        # save the transformed image
        basename = os.path.basename(image_path)
        cv2.imwrite(OUTPUT_DIR + '/' + basename, thresh)
        print("Proccessed " + basename)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Flag for manually verifying and/or setting document corners")

    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    interactive_mode = args["i"]

    scanner = DocScanner(interactive_mode)

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        scanner.scan(im_file_path)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        for im in im_files:
            scanner.scan(im_dir + '/' + im)

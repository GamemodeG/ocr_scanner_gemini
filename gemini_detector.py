#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document corner detection using Gemini Vision API
使用 Gemini Vision API 检测文档角点
Определение контуров документа с помощью Gemini Vision API

Author / 作者 / Автор: Selivanov Ivan PI51 / Селиванов Иван ПИ51

Uses Google Gemini 3 Flash to automatically detect 4 corners of a paper sheet.
使用 Google Gemini 3 Flash 自动检测纸张的4个角点。
Использует Google Gemini 3 Flash для автоматического определения 4 углов листа бумаги.

Install / 安装 / Установка: pip install --upgrade google-genai
"""

from google import genai
from google.genai import types
from PIL import Image
import json
import re
import numpy as np
import io


def detect_corners_with_gemini(image_path: str, api_key: str) -> np.ndarray | None:
    """
    Detects 4 corners using Gemini 3 Pro with Structured Output (JSON Schema)
    使用 Gemini 3 Pro 和结构化输出检测文档角点
    """
    client = genai.Client(api_key=api_key)
    
    image = Image.open(image_path)
    width, height = image.size
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    # Prompt emphasizing finding corners within strict bounds
    prompt = f"""Look at this image. Detect the four corners of the primary document or paper sheet visible in the photo. 
    Return the precise (x, y) pixel coordinates for the top_left, top_right, bottom_right, and bottom_left corners.
    The coordinates must be strictly within the image dimensions: {width}x{height}."""

    # Schema for strict JSON output
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "top_left": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            "top_right": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            "bottom_right": {"type": "ARRAY", "items": {"type": "INTEGER"}},
            "bottom_left": {"type": "ARRAY", "items": {"type": "INTEGER"}}
        },
        "required": ["top_left", "top_right", "bottom_right", "bottom_left"]
    }

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                thinking_config=types.ThinkingConfig(
                    thinking_level="HIGH"
                ),
            ),
        )
        
        # Parse standard JSON response
        try:
            corners_data = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"Gemini: Could not parse response: {response.text}")
            return None

        # Form coordinates array
        corners = np.array([
            corners_data["top_left"],
            corners_data["top_right"],
            corners_data["bottom_right"],
            corners_data["bottom_left"]
        ], dtype=np.float32)
        
        print(f"Gemini 3 Pro: structured corners found / 结构化角点 / структурные углы")
        print(f"  TL: {corners[0]}, TR: {corners[1]}")
        print(f"  BR: {corners[2]}, BL: {corners[3]}")
        
        return corners
        
    except Exception as e:
        print(f"Gemini: API error / API错误 / ошибка API - {e}")
        return None


def extract_text_with_gemini(image_path: str, api_key: str) -> dict | None:
    """
    Extracts text from document image using Gemini Vision API.
    Returns structured response with 3 sections: ASCII diagram, Markdown text, Description.
    使用 Gemini Vision API 从文档图像中提取文本，返回结构化响应（ASCII图、Markdown文本、描述）
    Извлекает текст из изображения с помощью Gemini Vision API.
    Возвращает структурированный ответ: ASCII схема, Markdown текст, Описание.
    """
    client = genai.Client(api_key=api_key)
    
    image = Image.open(image_path)
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    prompt = """Подробно проанализируй изображение и верни структурированный ответ с ЧЕТЫРЬМЯ отдельными разделами:

### РАЗДЕЛ 1: ASCII_DIAGRAM (ASCII представление изображения)
Каждую картинку, схему, диаграмму, таблицу подробно описывай и воссоздавай в ASCII!
Ссылки на иллюстрации не работают - ВОССОЗДАЙ их как ASCII-арт.

ФОРМАТ для КАЖДОГО элемента:
1. ASCII схема (используя символы: ┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼ → ← ↑ ↓ ● ○ ■ □ ▲ ▼ ═ ║ ╔ ╗ ╚ ╝)
2. Название схемы/картинки
3. Описание с буллетами:
   - Все детали и элементы
   - Все цифры и значения
   - Все связи между элементами
   - Что изображено и зачем

ПРАВИЛА:
- Воссоздай ВСЕ иллюстрации, графики, схемы, таблицы как ASCII-арт
- Покажи layout/расположение всех блоков на странице
- Под каждой схемой - название, потом буллеты с описанием
- Пиши ВСЕ числа, метки, подписи с оригинала
- Разделяй разные схемы пустой строкой
- Сохраняй пропорции и структуру оригинала

### РАЗДЕЛ 2: MARKDOWN_TEXT (Текст в формате Markdown)
Извлеки ВЕСЬ текстовый контент и отформатируй в Markdown:
- # ## ### для заголовков и разделов
- **жирный** для важных терминов и ключевых слов
- *курсив* для выделения
- Маркированные и нумерованные списки где уместно
- Таблицы в формате Markdown если есть табличные данные
- Формулы и уравнения
- Сохраняй все цифры, даты, числа точно
- НЕ пропускай ничего - весь видимый текст должен быть включён

### РАЗДЕЛ 3: DESCRIPTION (Описание изображения)
Подробное описание изображения с буллетами:
- Тип документа/изображения (фото, скан, схема, график и т.д.)
- Основная тема и назначение
- Описание всех визуальных элементов (цвета, формы, расположение)
- Связи между элементами на схемах
- Все числа, метки, подписи
- Контекст и общее впечатление
- Качество изображения

### РАЗДЕЛ 4: SEO_KEYWORDS (SEO и семантика для поиска)
Создай данные для поиска изображения:
- **Теги**: 10-20 ключевых слов/тегов через запятую (предмет, тема, объекты, действия)
- **Категории**: 3-5 категорий к которым относится изображение
- **Синонимы**: альтернативные названия и формулировки темы
- **Связанные понятия**: смежные темы и концепции
- **Alt-текст**: краткое описание для SEO (до 150 символов)
- **Заголовок**: привлекательный заголовок для изображения
- **Эмоции/настроение**: какие эмоции вызывает изображение
- **Целевая аудитория**: кому будет полезно/интересно

ВАЖНО: Отвечай ТОЛЬКО в JSON формате согласно заданной схеме."""

    # Schema for structured JSON output with 4 sections
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "ascii_diagram": {
                "type": "STRING",
                "description": "ASCII representation of the entire image layout"
            },
            "markdown_text": {
                "type": "STRING", 
                "description": "All text content extracted and formatted as Markdown"
            },
            "description": {
                "type": "STRING",
                "description": "Detailed description of the image with bullet points"
            },
            "seo_keywords": {
                "type": "STRING",
                "description": "SEO keywords, tags, categories, synonyms for image search"
            }
        },
        "required": ["ascii_diagram", "markdown_text", "description", "seo_keywords"]
    }

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        
        # Parse JSON response
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"Gemini OCR: Could not parse JSON response / JSON解析错误 / Ошибка парсинга JSON")
            # Fallback: return as plain text in markdown_text field
            return {
                "ascii_diagram": "",
                "markdown_text": response.text,
                "description": ""
            }
        
        # Second pass: Validate and fix ASCII representation
        ascii_content = result.get('ascii_diagram', '')
        if ascii_content and len(ascii_content) > 10:
            validation_prompt = f"""Проверь и ИСПРАВЬ следующее ASCII представление:

```
{ascii_content}
```

КРИТИЧЕСКИ ВАЖНО - ВЫРАВНИВАНИЕ:
1. ВСЕ строки в рамке должны быть ОДИНАКОВОЙ ДЛИНЫ
2. Правая граница (│) должна быть СТРОГО на одной позиции во всех строках
3. Добавь пробелы справа от текста ДО правой границы чтобы выровнять
4. Верхняя (┌───┐) и нижняя (└───┘) рамки должны быть той же ширины

ПРИМЕР ПРАВИЛЬНОГО ВЫРАВНИВАНИЯ:
┌────────────────────┐
│ Короткий текст     │
│ Текст подлиннее    │
│ Ещё текст          │
└────────────────────┘

ЗАДАЧИ:
1. Выровняй правую границу - все │ справа на одной позиции
2. Исправь сломанные углы и соединения
3. Ширина не более 60 символов
4. Не меняй содержимое, только выравнивание

Верни ТОЛЬКО исправленный ASCII-арт, без объяснений."""

            try:
                validation_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[validation_prompt],
                )
                
                fixed_ascii = validation_response.text.strip()
                # Remove markdown code block markers if present
                if fixed_ascii.startswith('```'):
                    lines = fixed_ascii.split('\n')
                    # Remove first line (```...) and last line (```)
                    if len(lines) > 2:
                        fixed_ascii = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
                
                if fixed_ascii:
                    result['ascii_diagram'] = fixed_ascii
                    print(f"Gemini OCR: ASCII validated and fixed / ASCII验证并修复 / ASCII проверен и исправлен")
            except Exception as e:
                print(f"Gemini OCR: ASCII validation skipped - {e}")
        
        print(f"Gemini OCR: structured extraction complete / 结构化提取完成 / структурное извлечение завершено")
        print(f"  - ASCII diagram: {len(result.get('ascii_diagram', ''))} chars")
        print(f"  - Markdown text: {len(result.get('markdown_text', ''))} chars")
        print(f"  - Description: {len(result.get('description', ''))} chars")
        
        return result
        
    except Exception as e:
        print(f"Gemini OCR: error / 错误 / ошибка - {e}")
        return None


def extract_text_with_gemini_simple(image_path: str, api_key: str) -> str | None:
    """
    Simple text extraction (legacy function for backward compatibility)
    简单文本提取（向后兼容的旧函数）
    Простое извлечение текста (устаревшая функция для обратной совместимости)
    """
    result = extract_text_with_gemini(image_path, api_key)
    if result:
        # Combine all sections into one string
        parts = []
        if result.get('ascii_diagram'):
            parts.append("## ASCII Схема\n\n```\n" + result['ascii_diagram'] + "\n```")
        if result.get('markdown_text'):
            parts.append("## Текст\n\n" + result['markdown_text'])
        if result.get('description'):
            parts.append("## Описание\n\n" + result['description'])
        return "\n\n---\n\n".join(parts) if parts else None
    return None


if __name__ == "__main__":
    # Test / 测试 / Тест
    import sys
    
    try:
        from config import GEMINI_API_KEY
    except ImportError:
        print("Error: create config.py with GEMINI_API_KEY / 错误：创建包含GEMINI_API_KEY的config.py / Ошибка: создайте config.py с вашим GEMINI_API_KEY")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage / 用法 / Использование: python gemini_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    corners = detect_corners_with_gemini(image_path, GEMINI_API_KEY)
    
    if corners is not None:
        print(f"\nCorner coordinates / 角点坐标 / Координаты углов:")
        print(corners)

"""
Text Extractors - Implementations of ITextExtractor

Извлекает текстовый контент из изображений документов с помощью AI.
Использует параллельные запросы для ускорения обработки.
"""

import json
import io
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from ..domain.interfaces import ITextExtractor
from ..domain.entities import TextExtractionResult


class GeminiTextExtractor(ITextExtractor):
    """
    Gemini Vision API-based text extraction с параллельными запросами.
    
    Разбивает один сложный запрос на 4 параллельных для ускорения:
    - ASCII диаграмма
    - Markdown текст  
    - Описание изображения
    - SEO ключевые слова
    """
    
    def __init__(self, api_key: str):
        """
        Инициализация Gemini экстрактора.
        
        Args:
            api_key: Google Gemini API key
        """
        self._api_key = api_key
        self._client = None
    
    @property
    def name(self) -> str:
        return "Gemini OCR"
    
    def _get_client(self):
        """Ленивая инициализация Gemini клиента."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client
    
    def _load_image_bytes(self, image_path: str) -> bytes:
        """Загрузить изображение и конвертировать в bytes."""
        from PIL import Image
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return buffered.getvalue()
    
    def _extract_ascii(self, image_bytes: bytes) -> str:
        """
        Извлечь ASCII диаграмму из изображения.
        
        Args:
            image_bytes: Изображение в формате JPEG bytes
            
        Returns:
            ASCII представление изображения
        """
        from google.genai import types
        
        prompt = """Создай ASCII представление изображения.

ФОРМАТ:
1. ASCII схема (используя символы: ┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼ → ← ↑ ↓ ● ○ ■ □ ▲ ▼ ═ ║ ╔ ╗ ╚ ╝)
2. Название схемы/картинки
3. Описание с буллетами

Каждую картинку, схему, диаграмму, таблицу подробно опиши и воссоздай в ASCII!
Верни только текст без JSON обёртки."""
        
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return response.text.strip()
        except Exception as e:
            print(f"ASCII extraction error: {e}")
            return ""
    
    def _extract_markdown(self, image_bytes: bytes) -> str:
        """
        Извлечь Markdown текст из изображения.
        
        Args:
            image_bytes: Изображение в формате JPEG bytes
            
        Returns:
            Текст в формате Markdown
        """
        from google.genai import types
        
        prompt = """Извлеки ВЕСЬ текстовый контент и отформатируй в Markdown:
- # ## ### для заголовков и разделов
- **жирный** для важных терминов
- Маркированные и нумерованные списки
- Таблицы в формате Markdown если есть табличные данные

Верни только Markdown текст без обёрток."""
        
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return response.text.strip()
        except Exception as e:
            print(f"Markdown extraction error: {e}")
            return ""
    
    def _extract_description(self, image_bytes: bytes) -> str:
        """
        Извлечь описание изображения.
        
        Args:
            image_bytes: Изображение в формате JPEG bytes
            
        Returns:
            Подробное описание изображения
        """
        from google.genai import types
        
        prompt = """Создай подробное описание изображения с буллетами:
- Тип документа/изображения
- Основная тема и назначение
- Описание всех визуальных элементов
- Контекст и общее впечатление

Верни только описание без обёрток."""
        
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return response.text.strip()
        except Exception as e:
            print(f"Description extraction error: {e}")
            return ""
    
    def _extract_seo(self, image_bytes: bytes) -> str:
        """
        Извлечь SEO ключевые слова.
        
        Args:
            image_bytes: Изображение в формате JPEG bytes
            
        Returns:
            SEO теги, категории и alt-текст
        """
        from google.genai import types
        
        prompt = """Создай SEO и семантику для поиска:
- **Теги**: 10-20 ключевых слов через запятую
- **Категории**: 3-5 категорий
- **Alt-текст**: краткое описание (до 150 символов)
- **Заголовок**: привлекательный заголовок

Верни форматированный текст без JSON."""
        
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return response.text.strip()
        except Exception as e:
            print(f"SEO extraction error: {e}")
            return ""
    
    def extract(self, image_path: str) -> Optional[TextExtractionResult]:
        """
        Извлечь текст через 4 ПАРАЛЛЕЛЬНЫХ запроса к Gemini.
        
        Использует ThreadPoolExecutor для одновременного выполнения:
        - ASCII диаграмма
        - Markdown текст
        - Описание
        - SEO ключевые слова
        
        Args:
            image_path: Путь к изображению документа
            
        Returns:
            TextExtractionResult со всеми извлечёнными данными
        """
        try:
            # Загружаем изображение один раз
            image_bytes = self._load_image_bytes(image_path)
            
            # Параллельные запросы через ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'ascii': executor.submit(self._extract_ascii, image_bytes),
                    'markdown': executor.submit(self._extract_markdown, image_bytes),
                    'description': executor.submit(self._extract_description, image_bytes),
                    'seo': executor.submit(self._extract_seo, image_bytes),
                }
                results = {k: f.result() for k, f in futures.items()}
            
            return TextExtractionResult(
                ascii_diagram=results['ascii'],
                markdown_text=results['markdown'],
                description=results['description'],
                seo_keywords=results['seo']
            )
            
        except Exception as e:
            print(f"Gemini parallel extraction error: {e}")
            return None


# OCR Scanner - Clean Architecture

## Принципы

### 1. Single Responsibility Principle (SRP)
Каждый файл/класс отвечает за одну вещь:
- `entities.py` - только определения сущностей
- `interfaces.py` - только абстрактные контракты
- `image_processor.py` - только операции с изображениями
- `corner_detectors.py` - только детекция углов

### 2. Separation of Concerns (SoC)
Четкое разделение по слоям:
```
src/
├── domain/           # Бизнес-сущности (без зависимостей)
├── infrastructure/   # Внешние сервисы (OpenCV, Gemini)
├── application/      # Бизнес-логика (оркестрация)
└── presentation/     # HTTP роуты (Flask)
```

### 3. Dependency Inversion (DIP)
- Внутренние слои не зависят от внешних
- Domain определяет интерфейсы (ICornerDetector, IImageProcessor)
- Infrastructure реализует эти интерфейсы
- Application использует интерфейсы, не конкретные классы

### 4. AI-Friendly Codebase
- Маленькие файлы (< 300 строк)
- Подробные docstrings
- Явные типы (type hints)
- Понятные имена

## Структура файлов

```
ocr_scanner/
├── app.py                 # Точка входа (composition root)
├── config.py              # Конфигурация API ключей
│
├── src/
│   ├── __init__.py        # Описание пакета
│   ├── container.py       # DI контейнер
│   │
│   ├── domain/            # СЛОЙ 1: Ядро (0 зависимостей)
│   │   ├── entities.py    # Corners, Document, ProcessingResult
│   │   └── interfaces.py  # ICornerDetector, IImageProcessor, ITextExtractor
│   │
│   ├── infrastructure/    # СЛОЙ 2: Внешние сервисы
│   │   ├── image_processor.py    # OpenCVImageProcessor
│   │   ├── corner_detectors.py   # OpenCVCornerDetector, GeminiCornerDetector
│   │   ├── text_extractors.py    # GeminiTextExtractor
│   │   └── file_manager.py       # FileManager
│   │
│   ├── application/       # СЛОЙ 3: Бизнес-логика
│   │   ├── scanner_service.py         # DocumentScannerService
│   │   └── text_extraction_service.py # TextExtractionService
│   │
│   └── presentation/      # СЛОЙ 4: HTTP
│       └── routes.py      # Flask blueprints
│
├── templates/             # HTML шаблоны
│   └── index.html
│
└── static/                # Статика
    ├── uploads/
    └── processed/
```

## Поток данных

```
HTTP Request
    ↓
[Presentation] routes.py → обрабатывает HTTP
    ↓
[Application] scanner_service.py → оркестрирует бизнес-логику
    ↓
[Infrastructure] image_processor.py → выполняет операции
    ↓
[Domain] entities.py → возвращает результат
    ↓
HTTP Response (JSON)
```

## Dependency Injection

Все зависимости инжектируются через `Container`:

```python
container = Container(gemini_api_key="...")

# Container создает и связывает все компоненты
scanner = container.opencv_scanner  # Уже с injected зависимостями
```

## Добавление нового функционала

### Новый детектор углов:
1. Создать класс в `infrastructure/corner_detectors.py`
2. Реализовать `ICornerDetector` интерфейс
3. Добавить в `container.py`

## GeminiCornerDetector — Алгоритм детекции через генерацию изображения

```
Оригинальное фото → Gemini 3 Pro Image → Фото с зелёными точками → OpenCV HSV → Координаты углов
```

1. Отправляем фото в `gemini-3-pro-image-preview` с промптом нарисовать зелёные точки на углах
2. Получаем сгенерированное изображение с точками-маркерами
3. Находим зелёные точки через HSV фильтрацию (OpenCV)
4. Масштабируем координаты к размеру оригинала
5. Упорядочиваем: top-left → top-right → bottom-right → bottom-left

### Новый эндпоинт:
1. Добавить route в `presentation/routes.py`
2. Создать сервис в `application/` если нужна бизнес-логика

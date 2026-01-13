# Document Scanner (OCR Preprocessing)

# 文档扫描器 (OCR 预处理)

# Сканер документов (OCR предобработка)

**Author / 作者 / Автор:** Selivanov Ivan PI51 / Селиванов Иван ПИ51

---

## Description / 描述 / Описание

Program for automatic document scanning:

- Detection of 4 corners of a paper sheet photographed at an angle
- Perspective Transform for alignment
- Binarization for improved text clarity
- **Optional:** Gemini 3 Flash AI for smart corner detection

自动文档扫描程序：

- 检测倾斜拍摄的纸张的4个角
- 透视变换进行校正
- 二值化以提高文字清晰度
- **可选:** Gemini 3 Flash AI 智能角点检测

Программа для автоматического сканирования документов:

- Определение 4 углов листа бумаги, снятого под углом
- Перспективное преобразование для выравнивания
- Бинаризация для повышения четкости текста
- **Опционально:** Gemini 3 Flash AI для умного определения углов

---

## Installation / 安装 / Установка

```bash
pip install opencv-python numpy google-genai pillow
```

---

## Usage / 使用方法 / Использование

### OpenCV Mode / OpenCV 模式 / Режим OpenCV

```bash
python document_scanner.py --image photo.jpg --debug
```

### Gemini AI Mode / Gemini AI 模式 / Режим Gemini AI

```bash
python document_scanner.py --image photo.jpg --gemini --debug
```

---

## Configuration / 配置 / Настройка

### Getting Gemini API Key / 获取 Gemini API 密钥 / Получение API ключа Gemini

1. Go to / 前往 / Перейдите на: https://console.cloud.google.com
2. Create a new project / 创建新项目 / Создайте новый проект
3. Go to APIs & Services → Credentials / 前往 API 和服务 → 凭据 / Перейдите в APIs & Services → Credentials
4. Create API Key / 创建 API 密钥 / Создайте API Key
5. Enable Generative Language API / 启用 Generative Language API / Включите Generative Language API

or use Google AI studio

Copy `config.example.py` to `config.py` and paste your key:
将 `config.example.py` 复制为 `config.py` 并粘贴您的密钥：
Скопируйте `config.example.py` в `config.py` и вставьте ваш ключ：

```python
GEMINI_API_KEY = "your_key_here"
```

---

## Algorithm / 算法流程 / Алгоритм

1. **Load image / 加载图像 / Загрузка изображения**
2. **Grayscale / 灰度转换 / Оттенки серого**
3. **Gaussian Blur / 高斯模糊 / Гауссово размытие**
4. **Edge Detection (Canny) / 边缘检测 / Детекция границ**
5. **Morphological Operations / 形态学操作 / Морфологические операции**
6. **Contour Detection / 轮廓检测 / Поиск контуров**
7. **Perspective Transform / 透视变换 / Перспективное преобразование**
8. **Sharpening / 锐化 / Повышение резкости**
9. **Binarization / 二值化 / Бинаризация**

---

## Output Files / 输出文件 / Выходные файлы

| File / 文件 / Файл | Description / 描述 / Описание                           |
| ---------------------- | --------------------------------------------------------------- |
| `*_scanned.jpg`      | Final result / 最终结果 / Финальный результат |
| `*_ALL_STAGES.jpg`   | All 9 stages grid / 所有9个阶段 / Все 9 этапов         |
| `*_1_original.jpg`   | Original / 原始 / Оригинал                              |
| `*_2_grayscale.jpg`  | Grayscale / 灰度 / Оттенки серого                  |
| `*_3_blurred.jpg`    | Blurred / 模糊 / Размытие                               |
| `*_4_edges.jpg`      | Edges / 边缘 / Границы                                   |
| `*_5_morphology.jpg` | Morphology / 形态学 / Морфология                      |
| `*_6_contour.jpg`    | Contour / 轮廓 / Контур                                   |
| `*_7_warped.jpg`     | Warped / 变换后 / После трансформации         |
| `*_8_sharpened.jpg`  | Sharpened / 锐化后 / После резкости                |
| `*_9_binary.jpg`     | Binary / 二值化 / Бинаризация                        |

---

## Dependencies / 依赖项 / Зависимости

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- google-genai (optional / 可选 / опционально)

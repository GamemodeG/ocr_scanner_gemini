# Configuration for API keys (TEMPLATE) / API密钥配置（模板）/ Конфигурация API ключей (ШАБЛОН)
# Copy this file as config.py and paste your keys / 将此文件复制为config.py并粘贴您的密钥 / Скопируйте этот файл как config.py и вставьте свои ключи

# ============================================================
# Google Gemini API Key
# ============================================================
# How to get the key / 如何获取密钥 / Как получить ключ:
# 1. Go to / 前往 / Перейдите на: https://console.cloud.google.com
# 2. Create a new project / 创建新项目 / Создайте новый проект
# 3. Remember the Project ID / 记住项目ID / Запомните Project ID
# 4. Go to APIs & Services → Credentials / 前往API和服务→凭据 / Перейдите в APIs & Services → Credentials
# 5. Create API Key / 创建API密钥 / Создайте API Key
# 6. Enable Generative Language API / 启用Generative Language API / Включите Generative Language API
#
# Install library / 安装库 / Установка библиотеки: pip install --upgrade google-genai
# ============================================================

GEMINI_API_KEY = ""

# Detection method / 检测方法 / Метод определения контуров: "opencv" or/或 "gemini"
DETECTION_METHOD = "gemini"

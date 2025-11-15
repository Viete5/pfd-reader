import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_ROOT_PATH = os.path.join(BASE_DIR, "chroma_db_users")

# RAG Settings
GIGA_MODEL_NAME = "GigaChat Lite"
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
EMBEDDING_MODEL = "cointegrated/rubert-tiny2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVER_K = 3
LLM_TEMPERATURE = 0.1

# Validation
def validate_config():
    """Проверяет наличие необходимых конфигураций"""
    missing = []
    if not GIGACHAT_AUTH_KEY:
        missing.append("GIGACHAT_AUTH_KEY")
    if not GIGACHAT_CLIENT_SECRET:
        missing.append("GIGACHAT_CLIENT_SECRET")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    
    if missing:
        raise ValueError(f"Отсутствуют обязательные переменные окружения: {', '.join(missing)}")

# Проверяем конфигурацию при импорте
validate_config()

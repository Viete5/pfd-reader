import os
from dotenv import load_dotenv
load_dotenv()
# keys
GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_CLIENT_SECRET = os.getenv("GIGACHAT_CLIENT_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PDF_PATH = os.path.join(BASE_DIR, "data", "Full.pdf")
# VECTOR_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
VECTOR_DB_ROOT_PATH = os.path.join(BASE_DIR, "chroma_db_users")
# settings of RAG
GIGA_MODEL_NAME = "GigaChat Lite"
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVER_K = 4
LLM_TEMPERATURE = 0.1
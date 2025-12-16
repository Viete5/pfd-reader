import os
import logging
import hashlib
import re
import shutil
import pdfplumber
from typing import Optional, List

logger = logging.getLogger(__name__)

# --- КОНФИГУРАЦИЯ ---
CACHE_DIR = os.path.join(os.getcwd(), "marker_cache")
PDF_CACHE_DIR = os.path.join(os.getcwd(), "pdf_cache")
VECTOR_DB_ROOT_PATH = os.path.join(os.getcwd(), "vector_db")  # Для совместимости

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# Попытка импорта Marker
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    logger.warning("⚠️ Marker-pdf не найден. Качество распознавания сложных формул может быть снижено.")

# Попытка импорта Embeddings (для совместимости, если нужно)
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
except ImportError:
    HuggingFaceEmbeddings = None

_MARKER_MODELS = None


def load_marker_models():
    """Загружает модели Marker один раз в память."""
    global _MARKER_MODELS
    if _MARKER_MODELS is None and MARKER_AVAILABLE:
        logger.info("📥 [Marker] Загрузка нейросетей... (Может занять время)")
        _MARKER_MODELS = create_model_dict()
    return _MARKER_MODELS


# --- ГЕОМЕТРИЧЕСКИЙ ПАРСЕР (План Б) ---
def parse_pdf_geometrically(file_path: str) -> str:
    """
    Парсит PDF, основываясь на координатах слов.
    Пытается "склеить" дроби, расположенные друг над другом.
    """
    logger.info("📐 Запуск Геометрического парсера (Fallback)...")
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for p_num, page in enumerate(pdf.pages):
                # Извлекаем слова с координатами
                words = page.extract_words(keep_blank_chars=True, x_tolerance=2, y_tolerance=3)

                # Группируем слова в строки по Y-координате (с допуском 8 пикселей)
                lines_map = {}
                for w in words:
                    y_key = int(w['top'] // 8) * 8
                    if y_key not in lines_map: lines_map[y_key] = []
                    lines_map[y_key].append(w)

                sorted_ys = sorted(lines_map.keys())
                processed_lines = []
                skip_y_indices = set()

                for i, y in enumerate(sorted_ys):
                    if i in skip_y_indices: continue

                    row = sorted(lines_map[y], key=lambda x: x['x0'])
                    row_text = " ".join([w['text'] for w in row])

                    # Координаты текущей строки
                    row_left = row[0]['x0']
                    row_right = row[-1]['x1']

                    # Проверка на дробь (есть ли строка прямо под этой?)
                    is_fraction = False
                    if i + 1 < len(sorted_ys):
                        next_y = sorted_ys[i + 1]
                        # Если следующая строка очень близко (меньше 18 пикселей)
                        if (next_y - y) < 18:
                            next_row = sorted(lines_map[next_y], key=lambda x: x['x0'])
                            next_text = " ".join([w['text'] for w in next_row])

                            next_left = next_row[0]['x0']
                            next_right = next_row[-1]['x1']

                            # Проверка перекрытия по горизонтали
                            overlap = min(row_right, next_right) - max(row_left, next_left)
                            if overlap > 5:  # Если перекрытие существенное
                                combined = f"({row_text}) / ({next_text})"
                                processed_lines.append(combined)
                                skip_y_indices.add(i + 1)  # Пропускаем следующую строку, так как объединили
                                is_fraction = True

                    if not is_fraction:
                        processed_lines.append(row_text)

                # Очистка текста от явного мусора (если кодировка совсем битая)
                clean_lines = []
                for line in processed_lines:
                    # Оставляем цифры, буквы, мат. символы
                    if len(line.strip()) > 1:
                        clean_lines.append(line)

                full_text.append(f"--- Page {p_num + 1} ---\n" + "\n".join(clean_lines))
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Fallback Parser Error: {e}")
        return "Error parsing PDF geometrically."


# --- ОСНОВНАЯ ФУНКЦИЯ ---
def extract_math_context_ultimate(file_path: str, force_refresh: bool = False) -> str:
    """
    Главная точка входа для получения текста PDF.
    Использует Marker (если есть) или геометрический парсер.
    """
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    # Добавляем размер файла в ID, чтобы отличать разные файлы с одним именем
    file_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path).replace(".pdf", "")

    cache_path = os.path.join(CACHE_DIR, f"{filename}_{file_hash}_{file_size}.md")

    # 1. Если не просили обновить принудительно - читаем кеш
    if os.path.exists(cache_path) and not force_refresh:
        logger.info(f"📖 Читаю из кеша: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    text_result = ""
    # 2. Попытка Marker
    if MARKER_AVAILABLE:
        logger.info(f"⚙️ [Marker] Запуск OCR для {filename}...")
        try:
            model_dict = load_marker_models()
            # Конфигурация для новых версий Marker (через словарь)
            config_dict = {
                "output_format": "markdown",
                "force_ocr": True,  # Принудительный OCR для битых PDF
                "languages": ["ru", "en"]
            }

            converter = PdfConverter(artifact_dict=model_dict, config=config_dict)
            rendered = converter(file_path)
            text_result, _, _ = text_from_rendered(rendered)
            logger.info("✅ Marker успешно отработал!")
        except Exception as e:
            logger.error(f"⚠️ Marker упал: {e}")
            logger.warning("🔄 Переключаюсь на Геометрический парсер...")
            text_result = parse_pdf_geometrically(file_path)
    else:
        text_result = parse_pdf_geometrically(file_path)

    # 3. Сохранение результата
    if text_result:
        # Если обновляли принудительно - удалим старые кеши этого файла
        if force_refresh:
            for f in os.listdir(CACHE_DIR):
                if f.startswith(filename) and f != os.path.basename(cache_path):
                    try:
                        os.remove(os.path.join(CACHE_DIR, f))
                    except:
                        pass
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text_result)

    return text_result


# --- ФУНКЦИИ СОВМЕСТИМОСТИ (для orchestrator.py) ---
def index_user_pdf(file_path: str, user_id: int) -> bool:
    """
    Вызывается при загрузке файла.
    Сохраняет файл и ПРИНУДИТЕЛЬНО обновляет индекс.
    """
    try:
        # 1. Очистка старых файлов этого пользователя
        # (Мы храним только 1 активный файл для решения задач, чтобы не путаться)
        for f in os.listdir(PDF_CACHE_DIR):
            if f.startswith(f"user_{user_id}"):
                try:
                    os.remove(os.path.join(PDF_CACHE_DIR, f))
                except:
                    pass

        # 2. Сохранение нового файла
        dest_path = os.path.join(PDF_CACHE_DIR, f"user_{user_id}.pdf")
        shutil.copy(file_path, dest_path)
        logger.info(f"📄 Файл сохранен для пользователя {user_id}: {dest_path}")

        # 3. Принудительный запуск парсера (Progress Bar в логах)
        logger.info(f"🔥 Индексация файла (это может занять 10-40 сек)...")
        extract_math_context_ultimate(dest_path, force_refresh=True)

        return True
    except Exception as e:
        logger.error(f"Index Error: {e}", exc_info=True)
        return False


def get_user_db_path(user_id: int) -> str:
    """Заглушка для совместимости путей"""
    return os.path.join(VECTOR_DB_ROOT_PATH, f"user_{user_id}")


def get_embeddings():
    """Заглушка для совместимости эмбеддингов"""
    if HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return None

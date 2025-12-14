# import os
# import tempfile
# from typing import List, Optional
# from abc import ABC, abstractmethod
# import pytesseract
# from PIL import Image, ImageEnhance, ImageFilter
# import cv2
# import numpy as np
# from langchain.schema import Document
# from langchain_community.document_loaders import PyPDFLoader
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class BaseDocumentLoader(ABC):
#     """Абстрактный базовый класс для загрузки документов"""
#
#     @abstractmethod
#     def load(self) -> List[Document]:
#         pass
#
#     @abstractmethod
#     def supports_format(self, file_path: str) -> bool:
#         pass
#
#
# class PDFLoader(BaseDocumentLoader):
#     """Загрузчик PDF файлов"""
#
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.loader = PyPDFLoader(file_path)
#
#     def load(self) -> List[Document]:
#         try:
#             return self.loader.load()
#         except Exception as e:
#             logger.error(f"Ошибка загрузки PDF: {e}")
#             raise
#
#     def supports_format(self, file_path: str) -> bool:
#         return file_path.lower().endswith('.pdf')
#
#
# class ImageLoader(BaseDocumentLoader):
#     """Загрузчик изображений с OCR"""
#
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
#
#     def _preprocess_image(self, image: Image.Image) -> Image.Image:
#         """Предобработка изображения для улучшения OCR"""
#         try:
#             # Конвертируем в numpy array для OpenCV обработки
#             img_array = np.array(image)
#
#             # Конвертируем в grayscale если нужно
#             if len(img_array.shape) == 3:
#                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#
#             # Убираем шум
#             img_array = cv2.medianBlur(img_array, 3)
#
#             # Повышаем контрастность
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             img_array = clahe.apply(img_array)
#
#             # Конвертируем обратно в PIL Image
#             return Image.fromarray(img_array)
#
#         except Exception as e:
#             logger.warning(f"Ошибка предобработки изображения: {e}")
#             return image
#
#     def _extract_text_from_image(self, image: Image.Image) -> str:
#         """Извлечение текста из изображения с помощью Tesseract"""
#         try:
#             # Предобработка
#             processed_image = self._preprocess_image(image)
#
#             # Конфигурация Tesseract для лучшего распознавания
#             custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя .,!?;:()-'
#
#             # Извлекаем текст
#             text = pytesseract.image_to_string(
#                 processed_image,
#                 lang='rus+eng',
#                 config=custom_config
#             )
#
#             # Пост-обработка текста
#             text = self._postprocess_text(text)
#
#             return text
#
#         except Exception as e:
#             logger.error(f"Ошибка OCR: {e}")
#             return ""
#
#     def _postprocess_text(self, text: str) -> str:
#         """Пост-обработка распознанного текста"""
#         lines = [line.strip() for line in text.split('\n') if line.strip()]
#
#         processed_lines = []
#         current_line = ""
#
#         for line in lines:
#             if len(line) < 50 and not line.endswith(('.', '!', '?')):
#                 current_line += " " + line if current_line else line
#             else:
#                 if current_line:
#                     processed_lines.append(current_line)
#                     current_line = ""
#                 processed_lines.append(line)
#
#         if current_line:
#             processed_lines.append(current_line)
#
#         return '\n'.join(processed_lines)
#
#     def load(self) -> List[Document]:
#         try:
#             with Image.open(self.file_path) as img:
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
#
#                 text = self._extract_text_from_image(img)
#
#                 if not text.strip():
#                     raise ValueError("Не удалось распознать текст на изображении")
#
#                 document = Document(
#                     page_content=text,
#                     metadata={
#                         "source": self.file_path,
#                         "file_type": "image",
#                         "page": 1
#                     }
#                 )
#
#                 return [document]
#
#         except Exception as e:
#             logger.error(f"Ошибка загрузки изображения: {e}")
#             raise
#
#     def supports_format(self, file_path: str) -> bool:
#         ext = os.path.splitext(file_path.lower())[1]
#         return ext in self.supported_formats
#
#
# class UniversalDocumentLoader:
#     """Универсальный загрузчик документов"""
#
#     def __init__(self):
#         self.loaders = [
#             PDFLoader(""),
#             ImageLoader("")
#         ]
#
#     def load_document(self, file_path: str) -> List[Document]:
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Файл не найден: {file_path}")
#
#         for loader_class in self.loaders:
#             loader = loader_class.__class__(file_path)
#             if loader.supports_format(file_path):
#                 logger.info(f"Используется загрузчик: {loader.__class__.__name__}")
#                 return loader.load()
#
#         raise ValueError(f"Неподдерживаемый формат файла: {file_path}")
#
#
# universal_loader = UniversalDocumentLoader()

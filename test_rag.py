import os
from src.tools.pdf_indexer import index_user_pdf

# Путь к вашему PDF
PDF_PATH = "C:/Users/Home/Desktop/document.pdf"  # ← ЗАМЕНИТЕ НА РЕАЛЬНЫЙ ПУТЬ
USER_ID = 12345

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ Файл не найден: {PDF_PATH}")
    else:
        print(f"📥 Индексация PDF для user_id={USER_ID}...")
        success = index_user_pdf(PDF_PATH, USER_ID)
        if success:
            print("✅ PDF успешно загружен и проиндексирован!")
        else:
            print("❌ Ошибка индексации")
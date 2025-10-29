import asyncio
from src.config import BASE_DIR
from src.core.orchestrator import handle_document_upload, handle_user_query
import os

# Формируем путь точно так же, как в config.py
TEST_PDF_PATH = os.path.join(BASE_DIR, "data", "Full.pdf")
USER_ID = 12345

async def main():
    print(f"📁 Путь к проекту: {BASE_DIR}")
    print(f"📄 Путь к PDF: {TEST_PDF_PATH}")

    if not os.path.exists(TEST_PDF_PATH):
        print("❌ ОШИБКА: Файл Full.pdf не найден!")
        print("Убедитесь, что у вас есть папка 'data' в корне проекта и внутри неё лежит 'Full.pdf'.")
        return

    # 1. Индексация
    print("\n🔍 Индексация PDF...")
    result = await handle_document_upload(USER_ID, TEST_PDF_PATH)
    print("→", result)

    if "успешно" not in result:
        print("❌ Индексация завершилась с ошибкой.")
        return

    # 2. Запрос
    answer = await handle_user_query(USER_ID, "What is this text about?")
    print("→ Ответ:", answer)

if __name__ == "__main__":
    asyncio.run(main())
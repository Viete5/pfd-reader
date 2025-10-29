
import asyncio
import logging
from src.tools.pdf_indexer import index_user_pdf
from src.agents.RAG import RAGAgent  # ← импортируем АГЕНТА

logger = logging.getLogger(__name__)

# Экземпляр агента — создаём ОДИН раз (он stateless)
_rag_agent = RAGAgent()

async def handle_document_upload(user_id: int, file_path: str) -> str:
    try:
        success: bool = await asyncio.to_thread(index_user_pdf, file_path, user_id)
        if success:
            return "✅ Ваш документ успешно проиндексирован! Теперь вы можете задавать вопросы по нему."
        else:
            return "❌ Не удалось обработать PDF."
    except Exception as e:
        logger.error(f"Ошибка при индексации (user_id={user_id}): {e}")
        return "❌ Внутренняя ошибка. Попробуйте позже."

async def handle_user_query(user_id: int, query: str) -> str:
    try:
        # Вызываем АГЕНТА через to_thread (его метод run — синхронный)
        response: str = await asyncio.to_thread(_rag_agent.run, user_id, query)
        return response
    except Exception as e:
        logger.error(f"Ошибка при запросе (user_id={user_id}): {e}")
        return "❌ Не удалось обработать запрос."

from src.tools.rag_query import UserRAGQuery

class RAGAgent:
    """
    Агент, отвечающий на вопросы по персональному PDF-документу пользователя.
    """

    def run(self, user_id: int, query: str) -> str:
        """
        Выполняет RAG-запрос для пользователя.
        Возвращает ответ или сообщение об ошибке.
        """
        try:
            rag = UserRAGQuery(user_id=user_id)
            return rag.ask(query)
        except FileNotFoundError:
            return "⚠️ У вас нет загруженного документа. Сначала отправьте PDF."
        except Exception as e:
            return "❌ Ошибка при генерации ответа."
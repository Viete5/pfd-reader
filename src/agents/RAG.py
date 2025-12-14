

from src.tools.rag_with_memory import UserRAGQuery
from typing import Dict


class RAGAgent:
    """
    Агент, отвечающий на вопросы по персональному PDF-документу пользователя.
    """


    def __init__(self):
        self.user_sessions: Dict[int, UserRAGQuery] = {}

    def _get_or_create_session(self, user_id: int) -> UserRAGQuery:
        """Получает активную сессию с памятью или создает новую."""
        if user_id not in self.user_sessions:
            # Создаем новый объект, который инициализирует цепь и память
            session = UserRAGQuery(user_id=user_id)
            self.user_sessions[user_id] = session
        return self.user_sessions[user_id]

    def run(self, user_id: int, query: str) -> str:
        """
        Выполняет RAG-запрос, используя сохраненную сессию с памятью.
        """
        try:
            # 1. Получаем объект с памятью
            rag = self._get_or_create_session(user_id)
            response = rag.ask(query)
            failure_keywords = [
                "не могу найти",
                "не содержится",
                "отсутствует информация",
                "не указано",
                "нет данных",  # Если ответ отсутствует
                "не нашел",
                "не нашёл",
                "отсутствуют сведения"
            ]

            # Проверяем, содержит ли ответ LLM хотя бы одну из фраз
            if any(keyword in response.lower() for keyword in failure_keywords):
                # Возвращаем специальный сигнал оркестратору
                return "NO_RAG_ANSWER"
            # 2. Отправляем вопрос в сохраненную цепь
            return response

        except FileNotFoundError:
            # Это произойдет, если _get_or_create_session не нашел базу
            return "⚠️ У вас нет загруженного документа. Сначала отправьте PDF."
        except Exception as e:
            # Логирование ошибки (для отладки)
            print(f"Ошибка в RAG с памятью для user {user_id}: {e}")
            return "❌ Ошибка при генерации ответа."

    def reset_session(self, user_id: int):
        """Закрывает и удаляет сессию RAG для пользователя."""
        if user_id in self.user_sessions:
            # 1. Получаем объект, чтобы вызвать метод close()
            session = self.user_sessions[user_id]

            # 2. Вызываем метод close() на загрузчике
            if hasattr(session, 'loader') and hasattr(session.loader, 'close'):
                session.loader.close()

            # 3. Удаляем сессию
            del self.user_sessions[user_id]
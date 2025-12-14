

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
    def get_note_text(self, user_id: int, max_chars: int = 15000) -> str:
        """
        Возвращает сырой текст конспекта пользователя для генерации квиза.
        Берёт текст из retriever внутри UserRAGQuery.
        """
        if user_id not in self.user_sessions:
            print("get_note_text: нет сессии для user_id", user_id)
            return ""

        try:
            session = self.user_sessions[user_id]
            retriever = getattr(session, "retriever", None)
            if retriever is None:
                print("get_note_text: у session нет поля retriever")
                return ""

            docs = retriever.get_relevant_documents("основные темы и понятия конспекта")
            if not docs:
                print("get_note_text: retriever вернул 0 документов")
                return ""

            chunks = []
            total = 0
            for doc in docs:
                text = getattr(doc, "page_content", "")
                if not text:
                    continue
                if total + len(text) > max_chars:
                    chunks.append(text[: max_chars - total])
                    break
                chunks.append(text)
                total += len(text)

            full_text = "\n\n".join(chunks)
            print("get_note_text: длина текста для квиза =", len(full_text))
            return full_text

        except Exception as e:
            print(f"Ошибка получения текста для квиза: {e}")
            return ""
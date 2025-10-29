import os

from functools import lru_cache
from langchain_gigachat.chat_models import GigaChat
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from src.config import EMBEDDING_MODEL, LLM_TEMPERATURE, RETRIEVER_K
from src.services.get_token import get_token
from src.tools.pdf_indexer import get_user_db_path


# ───────────────────────────────────────
# КЭШИРОВАНИЕ
# ───────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)





# ───────────────────────────────────────
# ОСНОВНОЙ КЛАСС — ОБРАБОТКА ЗАПРОСА ПОЛЬЗОВАТЕЛЯ
# ───────────────────────────────────────

class UserRAGQuery:
    def __init__(self, user_id: str | int):
        self.user_id = user_id
        self.user_db_path = get_user_db_path(user_id)

        # Проверяем: загружал ли пользователь PDF?
        if not os.path.exists(self.user_db_path):
            raise FileNotFoundError(f"Нет базы данных для пользователя {user_id}. Загрузите PDF.")

        # Инициализируем LLM (с кэшированным токеном!)
        self.llm = self._initialize_llm()

        # Получаем эмбеддинги из кэша
        self.embeddings = _get_embeddings()

        # Загружаем векторную БД пользователя (Chroma читает с диска — быстро)
        self.vectorstore = Chroma(
            persist_directory=self.user_db_path,
            embedding_function=self.embeddings
        )

        # Настраиваем ретривер (сколько чанков брать)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        # Собираем цепочку: вопрос → ретривер → LLM → ответ
        self.qa_chain = self._create_qa_chain()

    def _initialize_llm(self):
        """Создаёт GigaChat с валидным токеном."""
        token = get_token()
        if not token:
            raise ValueError("Не удалось получить Access Token.")
        return GigaChat(
            temperature=LLM_TEMPERATURE,
            verify_ssl_certs=False,
            access_token=token
        )

    def _create_qa_chain(self):
        """Создаёт стандартную RAG-цепочку 'stuff' (всё в один промпт)."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )

    def ask(self, question: str) -> str:
        """Отправляет вопрос в цепочку и возвращает ответ."""
        try:
            response = self.qa_chain.invoke({"query": question})
            return response.get('result', 'Не удалось получить ответ.')
        except Exception:
            return "Произошла ошибка при обработке вашего запроса."
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
# Используем наш новый загрузчик
from .rag_query import RAGLoader
from langchain.prompts import PromptTemplate

# Новый шаблон промпта для ConversationalRetrievalChain

# ... (Импорт LLM)

class UserRAGQuery:
    """
    Класс, который создает и хранит RAG-цепочку с памятью для одного пользователя.
    """

    def __init__(self, user_id: str | int):
        # 1. Загружаем компоненты
        loader = RAGLoader(user_id)
        llm, retriever = loader.get_components()

        # 2. Создание памяти с суммаризацией
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=3000,
            memory_key="chat_history",
            return_messages=True
        )

        # 3. Создание цепи RAG с памятью
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory

        )

    def ask(self, question: str) -> str:
        """Отправляет вопрос в цепочку с памятью и возвращает ответ."""
        response = self.qa_chain.invoke({"question": question})
        return response.get('answer', 'Не удалось получить ответ.')
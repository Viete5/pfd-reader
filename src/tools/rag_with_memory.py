from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
# Используем наш новый загрузчик
from .rag_query import RAGLoader
from langchain.prompts import PromptTemplate

# Новый шаблон промпта для ConversationalRetrievalChain

# ... (Импорт LLM)
from langchain.prompts import PromptTemplate

# 1. Промпт для сжатия запроса (важный шаг!)
CONDENSE_PROMPT = """
Учитывая историю разговора и последующий вопрос, твоя задача — сгенерировать автономный поисковый запрос.

**Правила:**
1.  Если последующий вопрос **полностью** понятен без контекста истории (например, "Что такое закон Ома?"), твой ответ — **ТОЛЬКО** этот вопрос. Игнорируй историю.
2.  Если последующий вопрос **зависит** от истории (например, "А где это применимо?"), используй историю, чтобы сформулировать **самостоятельный** запрос (например, "Где применим закон Ома?").

Твой ответ должен быть **ТОЛЬКО** автономным поисковым запросом, без приветствий или пояснений.

История чата:
{chat_history}
Последующий вопрос: {question}
Автономный поисковый запрос:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_PROMPT)

# 2. Промпт для финального ответа RAG
QA_PROMPT = """
Ты — интеллектуальный помощник для студентов, отвечающий на вопросы
**ИСКЛЮЧИТЕЛЬНО** на основе их конспектов. Твоя основная функция — извлекать
и объяснять информацию из предоставленного текста.

**Строгие Правила:**
1.  Ответь на ВОПРОС пользователя, используя **ТОЛЬКО** предоставленный ниже контекст из конспектов.
2.  Если контекст **НЕ содержит** ответа на вопрос (или информации недостаточно), ты **ДОЛЖЕН** честно ответить, что **не можешь найти эту информацию в предоставленных конспектах**.
3.  **НИКОГДА** не придумывай информацию и **НЕ используй** свои общие знания.
4.  Учитывай историю разговора, если это помогает понять текущий вопрос.

Контекст (из конспектов):
{context}

Вопрос: {question}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_PROMPT)

class UserRAGQuery:
    """
    Класс, который создает и хранит RAG-цепочку с памятью для одного пользователя.
    """

    def __init__(self, user_id: int):
        # 1. Загружаем компоненты
        self.loader = RAGLoader(user_id)
        llm, retriever = self.loader.get_components()

        # 2. Создание памяти с суммаризацией
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            memory_key="chat_history",
            return_messages=True
        )

        # 3. Создание цепи RAG с памятью
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs = {"prompt": QA_CHAIN_PROMPT},  # Для финального ответа
            condense_question_prompt = CONDENSE_QUESTION_PROMPT

        )

    def ask(self, question: str) -> str:
        """Отправляет вопрос в цепочку с памятью и возвращает ответ."""
        response = self.qa_chain.invoke({"question": question})
        return response.get('answer', 'Не удалось получить ответ.')
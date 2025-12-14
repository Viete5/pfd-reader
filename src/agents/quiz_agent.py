import json
import logging
from typing import Dict, Any

from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class QuizAgent:
    """
    Агент для генерации тестов по конспекту
    """

    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Инициализация GigaChat"""
        token = get_token()
        return GigaChat(
            temperature=0.1,
            verify_ssl_certs=False,
            access_token=token,
        )

    def generate_quiz(self, context_text: str, num_questions: int = 10, topic: str | None = None) -> Dict[str, Any]:
        """
        Генерирует список вопросов по тексту конспекта.
        Вызывается синхронно через asyncio.to_thread из orchestrator.py.
        """

        topic_hint = f"по теме: {topic}" if topic and topic != "весь" else "по основным темам конспекта"

        if not context_text:
            return {"questions": []}

        safe_context = context_text[:9000]

        prompt = f"""
Проанализируй следующий конспект и составь тест из {num_questions} вопросов {topic_hint}.

Текст конспекта:

\"\"\"{safe_context}\"\"\"

Требования:
1. Верни СТРОГО валидный JSON, без пояснений и без markdown.
2. Структура ответа:
{{
  "questions": [
    {{
      "question": "Текст вопроса?",
      "options": ["Вариант A", "Вариант B", "Вариант C", "Вариант D"],
      "correct_answer": "Текст правильного ответа",
      "explanation": "Краткое объяснение, почему этот ответ правильный"
    }}
  ]
}}
"""

        try:
            response = self.llm.invoke(prompt)
            return self._parse_quiz_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при генерации теста: {e}")
            return {"questions": []}

    def _parse_quiz_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ LLM в JSON-формат."""
        try:
            content = response.strip()

            if "```" in content:
                content = content.replace("```json", "").replace("```", "")

            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                content = content[start:end]

            data = json.loads(content)
            if "questions" not in data:
                return {"questions": []}
            return data

        except Exception as e:
            logger.warning(f"Ошибка парсинга JSON теста: {e}")
            return {"questions": []}






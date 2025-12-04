import logging
import json
from typing import Dict, Any, List

from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class QuizAgent:
    """
    Агент для генерации quiz из 10 вопросов по конспекту.
    """

    def __init__(self):
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Инициализация GigaChat (как в остальных агентах)."""
        token = get_token()
        return GigaChat(
            temperature=LLM_TEMPERATURE,
            verify_ssl_certs=False,
            access_token=token
        )

    def generate_quiz(self, context: str) -> Dict[str, Any]:
        """
        Генерирует quiz из 10 вопросов по тексту конспекта (context).
        """
        prompt = f"""
        Ты — преподаватель, который готовит обучающий тест по конспекту студента.

        Вот выдержка из конспекта:

        {context[:4000]}

        По этому тексту составь обучающий QUIZ из 10 вопросов.

        Требования:
        - Вопросы должны охватывать ключевые темы из конспекта.
        - Для КАЖДОГО вопроса дай 4 варианта ответа (A, B, C, D), только один правильный.
        - Можно использовать формулы и обозначения, если они явно следуют из текста.
        - Уровень сложности: средний, для самопроверки студента.
        - Отвечай ТОЛЬКО в виде валидного JSON-массива БЕЗ дополнительного текста до или после.
        Формат:

        [
          {{
            "question": "Текст вопроса?",
            "options": ["A. вариант", "B. вариант", "C. вариант", "D. вариант"],
            "correct": "A"
          }}
        ]
        """

        try:
            response = self.llm.invoke(prompt)
            questions = self._parse_quiz_response(response.content)

            # на всякий случай ограничим 10
            if len(questions) > 10:
                questions = questions[:10]

            return {
                "type": "quiz",
                "questions": questions
            }

        except Exception as e:
            logger.error(f"Ошибка при генерации quiz: {e}")
            return {"type": "quiz", "questions": []}

    def _parse_quiz_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Парсит JSON с вопросами, по тем же принципам, что у других агентов.
        """
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            logger.error(f"Ошибка парсинга quiz JSON: {e}")

        return []

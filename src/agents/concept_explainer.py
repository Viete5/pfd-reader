import re
import logging
from typing import List, Dict, Any
from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)

class ConceptExplainerAgent:
    """
    Агент для извлечения и объяснения ключевых концептов из текста
    """
    
    def __init__(self):
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Инициализация GigaChat"""
        token = get_token()
        return GigaChat(
            temperature=LLM_TEMPERATURE,
            verify_ssl_certs=False,
            access_token=token
        )
    
    def extract_concepts(self, text: str, max_concepts: int = 10) -> List[Dict[str, Any]]:
        """
        Извлекает ключевые концепты из текста
        """
        prompt = f"""
        Проанализируй следующий текст и выдели {max_concepts} самых важных концептов, терминов, теорий или методов.
        Для каждого концепта предоставь:
        - Название
        - Категория (физика, математика, программирование и т.д.)
        - Уровень сложности (базовый, средний, продвинутый)
        - Краткое определение (1-2 предложения)
        
        Текст для анализа:
        {text[:4000]}
        
        Верни ответ в формате:
        КОНЦЕПТ: [название]
        КАТЕГОРИЯ: [категория]
        УРОВЕНЬ: [уровень]
        ОПРЕДЕЛЕНИЕ: [определение]
        ---
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_concepts_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при извлечении концептов: {e}")
            return []
    
    def _parse_concepts_response(self, response: str) -> List[Dict[str, Any]]:

        """Парсит ответ LLM в структурированный формат"""
        concepts = []
        current_concept = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('КОНЦЕПТ:'):
                if current_concept:
                    concepts.append(current_concept)
                current_concept = {'name': line.replace('КОНЦЕПТ:', '').strip()}
            elif line.startswith('КАТЕГОРИЯ:') and current_concept:
                current_concept['category'] = line.replace('КАТЕГОРИЯ:', '').strip()
            elif line.startswith('УРОВЕНЬ:') and current_concept:
                current_concept['level'] = line.replace('УРОВЕНЬ:', '').strip()
            elif line.startswith('ОПРЕДЕЛЕНИЕ:') and current_concept:
                current_concept['definition'] = line.replace('ОПРЕДЕЛЕНИЕ:', '').strip()
            elif line == '---' and current_concept:
                concepts.append(current_concept)
                current_concept = {}
        
        if current_concept and 'name' in current_concept:
            concepts.append(current_concept)
        
        return concepts
    
    def explain_concept(self, concept: str, context: str = "") -> Dict[str, Any]:
        """
        Генерирует подробное объяснение концепта
        """
        prompt = f"""
        Подробно объясни концепт "{concept}" как студенту.
        {f"Контекст: {context}" if context else ""}
        
        Структура объяснения:
        1. Простое определение (что это?)
        2. Основные принципы и характеристики
        3. Практические примеры и применение
        4. Связи с другими концептами
        5. Возможные трудности в понимании и как их преодолеть
        
        Объяснение должно быть понятным, с примерами из реальной жизни.
        Используй аналогии для сложных моментов.
        
        "Верни ответ в формате JSON. Не включай никаких дополнительных комментариев или текста до или после JSON-объекта. Формат json-объекта:
        {{
            "explanation": "полное объяснение",
            "key_points": ["ключевой момент 1", "ключевой момент 2"],
            "examples": ["пример 1", "пример 2"],
            "study_tips": ["совет 1", "совет 2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_explanation_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при генерации объяснения: {e}")
            return {
                "explanation": f"Концепт '{concept}' - это важное понятие в изучаемой области.",
                "key_points": ["Основное понятие предмета", "Имеет практическое применение"],
                "examples": ["Пример из реальной жизни"],
                "study_tips": ["Изучите основные определения", "Практикуйтесь на примерах"]
            }
    
    def _parse_explanation_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ с объяснением"""

        try:
            # Ищем JSON в ответе
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                import json
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback если не удалось распарсить JSON
        return {}

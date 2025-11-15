import logging
from typing import List, Dict, Any
from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)

class StudyAdvisorAgent:
    """
    Агент для создания персонализированных учебных планов и рекомендаций
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
    
    def get_study_advice(self) -> Dict[str, Any]:
        """
        Предоставляет общие учебные советы
        """
        prompt = """
        Дай универсальные учебные советы для студентов. Включи:
        - Методы эффективного обучения
        - Техники запоминания
        - Советы по тайм-менеджменту
        - Рекомендации по отдыху и перерывам
        
        Верни в формате JSON:
        {
            "advice": "основной текст советов",
            "quick_tips": ["совет 1", "совет 2", "совет 3"],
            "methods": ["метод 1", "метод 2"]
        }
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_advice_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при получении советов: {e}")
            return self._get_default_advice()
    
    def get_notes_advice(self, context: str = "") -> Dict[str, Any]:
        """
        Советы по ведению конспектов
        """
        prompt = f"""
        Дай советы по эффективному ведению конспектов.
        {f"Контекст: {context}" if context else ""}
        
        Включи:
        - Методы структурирования заметок
        - Техники визуального оформления
        - Советы по быстрому конспектированию
        - Рекомендации по повторению
        
        Верни в формате JSON:
        {{
            "advice": "основные рекомендации",
            "techniques": ["техника 1", "техника 2"],
            "tools": ["инструмент 1", "инструмент 2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_advice_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при получении советов по конспектам: {e}")
            return {
                "advice": "Используйте четкую структуру, выделяйте ключевые моменты, регулярно повторяйте материал.",
                "techniques": ["Метод Корнелла", "Ментальные карты", "Цветовое кодирование"],
                "tools": ["Бумажные заметки", "Цифровые приложения", "Диктофон"]
            }
    
    def get_memory_techniques(self) -> Dict[str, Any]:
        """
        Техники запоминания
        """
        prompt = """
        Опиши эффективные техники запоминания информации для студентов.
        
        Верни в формате JSON:
        {
            "advice": "общие рекомендации",
            "techniques": ["техника 1", "техника 2"],
            "exercises": ["упражнение 1", "упражнение 2"]
        }
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_advice_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при получении техник запоминания: {e}")
            return {
                "advice": "Используйте интервальное повторение и ассоциации для лучшего запоминания.",
                "techniques": ["Интервальное повторение", "Мнемотехники", "Ассоциации"],
                "exercises": ["Карточки для повторения", "Пересказ материала", "Решение задач"]
            }
    
    def improve_notes(self, notes_sample: str) -> Dict[str, Any]:
        """
        Анализирует и улучшает конспекты
        """
        prompt = f"""
        Проанализируй следующий образец конспекта и дай рекомендации по улучшению:
        
        {notes_sample[:2000]}
        
        Обрати внимание на:
        - Структуру и организацию
        - Ясность изложения
        - Полноту охвата темы
        - Визуальное оформление
        
        Верни в формате JSON:
        {{
            "suggestions": "основные рекомендации",
            "structure_tips": ["совет 1", "совет 2"],
            "visual_improvements": ["улучшение 1", "улучшение 2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_advice_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при улучшении конспекта: {e}")
            return {
                "suggestions": "Добавьте четкие заголовки, используйте маркированные списки, выделяйте ключевые термины.",
                "structure_tips": ["Используйте иерархию заголовков", "Группируйте связанные concepts"],
                "visual_improvements": ["Цветовое выделение", "Диаграммы и схемы", "Отступы и пробелы"]
            }
    
    def create_study_plan(self, topic: str, timeframe: str, context: str = "") -> Dict[str, Any]:
        """
        Создает учебный план
        """
        prompt = f"""
        Создай учебный план по теме "{topic}" на период "{timeframe}".
        {f"Контекст: {context}" if context else ""}
        
        Включи:
        - Распределение по дням/неделям
        - Конкретные темы для каждого занятия
        - Рекомендуемые материалы
        - Практические задания
        
        Верни в формате JSON:
        {{
            "plan": [
                {{
                    "day": "День 1",
                    "focus": "Основная тема",
                    "materials": "Рекомендуемые материалы",
                    "tasks": "Задания"
                }}
            ],
            "recommendations": ["рекомендация 1", "рекомендация 2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return self._parse_advice_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при создании учебного плана: {e}")
            return self._get_default_study_plan(topic, timeframe)
    
    def _parse_advice_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ с советами"""
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
        
        # Fallback
        return {
            "advice": response,
            "quick_tips": ["Регулярно повторяйте материал", "Делайте перерывы", "Практикуйтесь"],
            "methods": ["Активное обучение", "Практическое применение"]
        }
    
    def _get_default_advice(self) -> Dict[str, Any]:
        """Возвращает советы по умолчанию"""
        return {
            "advice": "Регулярность важнее длительности занятий. Делайте перерывы каждые 45-50 минут.",
            "quick_tips": [
                "Занимайтесь регулярно",
                "Делайте перерывы", 
                "Повторяйте материал",
                "Практикуйтесь на задачах",
                "Объясняйте материал другим"
            ],
            "methods": [
                "Помодоро техника",
                "Интервальное повторение",
                "Активное recall"
            ]
        }
    
    def _get_default_study_plan(self, topic: str, timeframe: str) -> Dict[str, Any]:
        """Возвращает учебный план по умолчанию"""
        return {
            "plan": [
                {
                    "day": "День 1",
                    "focus": f"Основные понятия {topic}",
                    "materials": "Базовые учебники и лекции",
                    "tasks": "Изучить определения и основные принципы"
                },
                {
                    "day": "День 2", 
                    "focus": "Углубленное изучение",
                    "materials": "Специализированная литература",
                    "tasks": "Решать практические задачи"
                }
            ],
            "recommendations": [
                "Следуйте расписанию",
                "Регулярно повторяйте",
                "Практикуйтесь ежедневно"
            ]
        }

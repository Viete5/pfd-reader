import logging
from typing import List, Dict, Any
from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)

class SourceFinderAgent:
    """
    Агент для поиска релевантных учебных источников
    """
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.knowledge_bases = {
            "physics": {
                "textbooks": [
                    "Фейнмановские лекции по физике",
                    "Берклеевский курс физики", 
                    "Савельев - Курс общей физики"
                ],
                "online_courses": [
                    "Coursera - Физика",
                    "Stepik - Основы физики"
                ]
            },
            "mathematics": {
                "textbooks": [
                    "Высшая математика - Данко",
                    "Математический анализ - Фихтенгольц"
                ],
                "online_courses": [
                    "Coursera - Математика для физиков",
                    "MIT OpenCourseWare - Mathematics"
                ]
            }
        }
    
    def _initialize_llm(self):
        """Инициализация GigaChat"""
        token = get_token()
        return GigaChat(
            temperature=LLM_TEMPERATURE,
            verify_ssl_certs=False,
            access_token=token
        )
    
    def find_sources(self, topic: str, context: str = "") -> Dict[str, Any]:
        """
        Находит источники для конкретной темы
        """
        prompt = f"""
        Для темы "{topic}" предложи лучшие учебные ресурсы.
        {f"Контекст: {context}" if context else ""}
        
        Включи разные типы источников:
        - Учебники и книги
        - Онлайн-курсы
        - Научные статьи
        - Обучающие видео
        - Интерактивные платформы
        
        Для каждого источника укажи:
        - Название
        - Тип (учебник, онлайн-курс, статья, видео)
        - Уровень (начальный, средний, продвинутый)
        - Краткое описание
        - Где найти (ссылка или место)
        
        Верни ответ в формате JSON:
        {{
            "sources": [
                {{
                    "name": "Название",
                    "type": "Тип",
                    "level": "Уровень",
                    "description": "Описание",
                    "link": "Ссылка"
                }}
            ],
            "study_path": ["Этап 1", "Этап 2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_sources_response(response.content)
            
            # Добавляем источники из базы знаний
            base_sources = self._get_base_sources(topic)
            if base_sources:
                result["sources"] = base_sources + result.get("sources", [])
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при поиске источников: {e}")
            return self._get_fallback_sources(topic)
    
    def _parse_sources_response(self, response: str) -> Dict[str, Any]:
        """Парсит ответ с источниками"""
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
            "sources": [
                {
                    "name": f"Учебники по {topic}",
                    "type": "учебник",
                    "level": "разный",
                    "description": f"Классические учебники по теме",
                    "link": f"https://www.google.com/search?q={topic}+учебник"
                }
            ],
            "study_path": ["Изучите базовые понятия", "Перейдите к практическим задачам"]
        }
    
    def _get_base_sources(self, topic: str) -> List[Dict[str, str]]:
        """Возвращает источники из базы знаний"""
        sources = []
        
        # Простой матчинг по ключевым словам
        if any(word in topic.lower() for word in ['физик', 'механ', 'электр']):
            category = "physics"
        elif any(word in topic.lower() for word in ['математ', 'алгебр', 'геометр']):
            category = "mathematics"
        else:
            return sources
        
        category_data = self.knowledge_bases.get(category, {})
        
        for source_type, source_list in category_data.items():
            for source_name in source_list[:2]:
                sources.append({
                    'name': source_name,
                    'type': source_type,
                    'level': 'средний',
                    'description': f'Классический ресурс по {category}',
                    'link': self._generate_search_link(source_name, topic)
                })
        
        return sources
    
    def _get_fallback_sources(self, topic: str) -> Dict[str, Any]:
        """Возвращает базовые источники при ошибке"""
        return {
            "sources": [
                {
                    'name': f'Учебники по {topic}',
                    'type': 'учебник',
                    'level': 'разный',
                    'description': f'Классические учебники по теме "{topic}"',
                    'link': self._generate_search_link(topic, "учебник")
                },
                {
                    'name': 'Coursera',
                    'type': 'онлайн-курс', 
                    'level': 'начальный-средний',
                    'description': f'Курсы по теме "{topic}"',
                    'link': f'https://www.coursera.org/search?query={topic}'
                }
            ],
            "study_path": [
                "Начните с базовых понятий",
                "Изучите основные принципы", 
                "Практикуйтесь на задачах"
            ]
        }
    
    def _generate_search_link(self, source: str, topic: str) -> str:
        """Генерирует поисковую ссылку"""
        search_query = f"{source} {topic}"
        return f"https://www.google.com/search?q={search_query.replace(' ', '+')}"

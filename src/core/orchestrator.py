import asyncio
import logging
import os
import re
from typing import Dict, Any, List
from src.tools.pdf_indexer import index_user_pdf
from src.agents.RAG import RAGAgent
from src.agents.concept_explainer import ConceptExplainerAgent
from src.agents.source_finder import SourceFinderAgent
from src.agents.study_advisor import StudyAdvisorAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создаем экземпляры агентов
_rag_agent = RAGAgent()
_concept_explainer = ConceptExplainerAgent()
_source_finder = SourceFinderAgent()
_study_advisor = StudyAdvisorAgent()

async def handle_document_upload(user_id: int, file_path: str) -> str:
    """
    Обрабатывает загрузку конспекта студента
    """
    try:
        logger.info(f"📥 Обработка конспекта для студента {user_id}")
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        if file_size > 10:
            return "❌ Файл слишком большой. Максимальный размер - 10MB."
        
        success: bool = await asyncio.to_thread(index_user_pdf, file_path, user_id)
        
        if success:
            logger.info(f"✅ Конспект студента {user_id} успешно обработан")
            return """✅ Ваш конспект успешно обработан!

Теперь я могу помочь вам:

• 🧠 **Объяснить понятия** - "объясни что такое [термин]"
• 📚 **Найти источники** - "найди материалы по [теме]"  
• 🎯 **Дать учебные советы** - "как лучше учить?"
• 📝 **Улучшить конспекты** - "как улучшить мои заметки"
• 🔍 **Ответить на вопросы** - любые вопросы по конспекту

Задавайте вопросы по вашему конспекту!"""
        else:
            logger.error(f"❌ Не удалось обработать конспект для студента {user_id}")
            return "❌ Не удалось обработать файл. Убедитесь, что это читаемый PDF."
            
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке конспекта (user_id={user_id}): {e}")
        return "❌ Произошла ошибка при обработке файла."

async def handle_user_query(user_id: int, query: str) -> str:
    """
    Обрабатывает запросы студентов с использованием специализированных агентов
    """
    try:
        logger.info(f"💬 Запрос от студента {user_id}: {query}")
        
        # Простые команды
        if query.lower() in ['/start', '/help', 'помощь', 'help']:
            return get_help_message()
        
        # Определяем тип запроса и направляем к соответствующему агенту
        if any(word in query.lower() for word in ['объясни', 'что такое', 'поясни', 'расскажи про']):
            return await _handle_concept_explanation(user_id, query)
        
        elif any(word in query.lower() for word in ['найди', 'источник', 'материал', 'литератур', 'книг', 'учебник']):
            return await _handle_source_finding(user_id, query)
        
        elif any(word in query.lower() for word in ['совет', 'как учить', 'метод', 'учебные', 'изучать', 'подход']):
            return await _handle_study_advice(user_id, query)
        
        elif any(word in query.lower() for word in ['улучши', 'структур', 'оформи', 'конспект', 'заметк']):
            return await _handle_notes_improvement(user_id, query)

        elif any(word in query.lower() for word in ['план', 'расписание', 'график', 'изучен']):
            return await _handle_study_plan(user_id, query)
        
        else:
            # Общие вопросы - используем RAG
            return await asyncio.to_thread(_rag_agent.run, user_id, query)
        
    except FileNotFoundError:
        return "⚠️ Сначала загрузите конспект! Используйте команду /start для помощи."
    
    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}")
        return "❌ Произошла ошибка. Попробуйте переформулировать вопрос."

async def _handle_concept_explanation(user_id: int, query: str) -> str:
    """Обработка запросов на объяснение понятий"""
    try:
        # Получаем контекст из конспекта
        context = await _get_context_from_notes(user_id, query)
        
        # Извлекаем концепт из запроса
        concept = _extract_concept_from_query(query)
        
        if not concept:
            return "❌ Не смог определить, какое понятие объяснить. Попробуйте: 'Объясни что такое [понятие]'"
        
        # Получаем объяснение от агента
        explanation_result = await asyncio.to_thread(
            _concept_explainer.explain_concept, 
            concept, 
            context
        )
        
        if explanation_result and "explanation" in explanation_result:

            response = f"🧠 **Объяснение: {concept}**\n\n"
            response += explanation_result["explanation"]
            
            if "key_points" in explanation_result:
                response += f"\n\n🔑 **Ключевые моменты:**\n"
                for point in explanation_result["key_points"][:3]:
                    response += f"• {point}\n"
            
            if "examples" in explanation_result:
                response += f"\n💡 **Примеры:**\n"
                for example in explanation_result["examples"][:2]:
                    response += f"• {example}\n"
            
            return response
        else:
            # Если агент не смог объяснить, используем RAG
            return await asyncio.to_thread(_rag_agent.run, user_id, f"Объясни понятие: {concept}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка объяснения: {e}")
        return f"❌ Не удалось объяснить понятие. Попробуйте задать вопрос по-другому."

async def _handle_source_finding(user_id: int, query: str) -> str:
    """Обработка запросов на поиск источников"""
    try:
        context = await _get_context_from_notes(user_id, query)
        topic = _extract_topic_from_query(query)
        
        if not topic:
            return "❌ Не смог определить тему для поиска. Попробуйте: 'Найди материалы по [теме]'"
        
        # Получаем источники от агента
        sources_result = await asyncio.to_thread(
            _source_finder.find_sources,
            topic,
            context
        )
        
        if sources_result and "sources" in sources_result:
            response = f"📚 **Материалы по теме: {topic}**\n\n"
            
            # Группируем источники по типам
            sources_by_type = {}
            for source in sources_result["sources"]:
                source_type = source.get("type", "разное")
                if source_type not in sources_by_type:
                    sources_by_type[source_type] = []
                sources_by_type[source_type].append(source)
            
            # Формируем ответ
            for source_type, sources in sources_by_type.items():
                response += f"**{source_type.upper()}:**\n"
                for source in sources[:3]:  # Ограничиваем 3 источниками на тип
                    response += f"• **{source['name']}**"
                    if source.get('description'):
                        response += f" - {source['description']}"
                    if source.get('link'):
                        response += f"\n  🔗 {source['link']}"
                    response += "\n"
                response += "\n"
            
            if "study_path" in sources_result:
                response += f"🎯 **Рекомендуемый порядок изучения:**\n"
                for stage in sources_result["study_path"][:3]:
                    response += f"• {stage}\n"
            
            return response
        else:
            return "❌ Не удалось найти подходящие источники по этой теме."
            
    except Exception as e:
        logger.error(f"❌ Ошибка поиска источников: {e}")
        return "❌ Произошла ошибка при поиске материалов. Попробуйте другую тему."

async def _handle_study_advice(user_id: int, query: str) -> str:
    """Обработка запросов на учебные советы"""
    try:
        # Определяем тип совета
        if any(word in query.lower() for word in ['конспект', 'заметк', 'запис']):
            # Советы по ведению конспектов
            notes_context = await _get_context_from_notes(user_id, "конспект методика")
            advice_result = await asyncio.to_thread(
                _study_advisor.get_notes_advice,
                notes_context
            )
            
            if advice_result and "advice" in advice_result:
                response = "📝 **Советы по ведению конспектов:**\n\n"
                response += advice_result["advice"]
                
                if "techniques" in advice_result:
                    response += f"\n🎯 **Эффективные методики:**\n"
                    for technique in advice_result["techniques"][:4]:
                        response += f"• {technique}\n"
                
                return response
        
        elif any(word in query.lower() for word in ['запоминан', 'памят', 'повторен']):
            # Советы по запоминанию
            advice_result = await asyncio.to_thread(_study_advisor.get_memory_techniques)
            
        else:
            # Общие учебные советы
            advice_result = await asyncio.to_thread(_study_advisor.get_study_advice)
        
        if advice_result and "advice" in advice_result:
            response = "🎓 **Учебные советы:**\n\n"
            response += advice_result["advice"]
            
            if "quick_tips" in advice_result:
                response += f"\n💡 **Быстрые советы:**\n"
                for tip in advice_result["quick_tips"][:5]:
                    response += f"• {tip}\n"
            
            return response
        else:
            return "❌ Не удалось получить учебные советы."
        
    except Exception as e:
        logger.error(f"❌ Ошибка предоставления советов: {e}")
        return "❌ Произошла ошибка при получении советов."

async def _handle_notes_improvement(user_id: int, query: str) -> str:
    """Обработка запросов на улучшение конспектов"""
    try:
        # Получаем пример конспекта пользователя
        notes_sample = await _get_context_from_notes(user_id, "конспект структура")
        
        if not notes_sample:
            return "❌ Не найдено конспектов для анализа. Сначала загрузите свой конспект."
        
        improvement_result = await asyncio.to_thread(
            _study_advisor.improve_notes,
            notes_sample
        )
        
        if improvement_result and "suggestions" in improvement_result:
            response = "✨ **Рекомендации по улучшению конспекта:**\n\n"
            response += improvement_result["suggestions"]
            
            if "structure_tips" in improvement_result:
                response += f"\n🏗 **Советы по структуре:**\n"
                for tip in improvement_result["structure_tips"][:3]:
                    response += f"• {tip}\n"
            
            if "visual_improvements" in improvement_result:
                response += f"\n🎨 **Визуальное оформление:**\n"
                for improvement in improvement_result["visual_improvements"][:3]:
                    response += f"• {improvement}\n"
            
            return response
        else:
            return "❌ Не удалось проанализировать конспект."
            
    except Exception as e:
        logger.error(f"❌ Ошибка улучшения конспекта: {e}")
        return "❌ Произошла ошибка при анализе конспекта."

async def _handle_study_plan(user_id: int, query: str) -> str:
    """Обработка запросов на создание учебного плана"""
    try:
        # Извлекаем тему и сроки из запроса
        topic = _extract_topic_from_query(query)
        timeframe = _extract_timeframe_from_query(query)
        
        # Получаем контекст по теме
        context = await _get_context_from_notes(user_id, topic or "учебный план")
        
        plan_result = await asyncio.to_thread(
            _study_advisor.create_study_plan,
            topic or "учебный материал",
            timeframe or "1 неделя",
            context
        )
        
        if plan_result and "plan" in plan_result:
            response = f"📅 **Учебный план{' по ' + topic if topic else ''}**\n"
            response += f"⏱ Срок: {timeframe or '1 неделя'}\n\n"
            
            for i, day_plan in enumerate(plan_result["plan"][:7], 1):  # Ограничиваем неделей
                response += f"**День {i}:**\n"
                response += f"🎯 {day_plan.get('focus', 'Основные темы')}\n"
                response += f"📚 {day_plan.get('materials', 'Рекомендуемые материалы')}\n"
                response += f"✅ {day_plan.get('tasks', 'Задания')}\n\n"
            
            if "recommendations" in plan_result:
                response += "💡 **Рекомендации:**\n"
                for rec in plan_result["recommendations"][:3]:
                    response += f"• {rec}\n"
            
            return response
        else:
            return "❌ Не удалось создать учебный план."
            
    except Exception as e:
        logger.error(f"❌ Ошибка создания плана: {e}")
        return "❌ Произошла ошибка при создании учебного плана."

async def _get_context_from_notes(user_id: int, query: str) -> str:
    """Получает релевантный контекст из конспектов пользователя"""
    try:
        # Используем RAG агента для получения релевантных фрагментов
        context_query = f"Контекст для: {query}"
        context_response = await asyncio.to_thread(_rag_agent.run, user_id, context_query)
        
        # Если ответ слишком длинный, обрезаем его
        if len(context_response) > 1000:
            return context_response[:1000] + "..."
        return context_response
        
    except Exception as e:
        logger.warning(f"Не удалось получить контекст: {e}")
        return ""

def _extract_concept_from_query(query: str) -> str:
    """Извлекает понятие из запроса"""
    patterns = [
        r'объясни\s+(?:что\s+такое\s+)?(.+?)(?:\?|$|\.)',
        r'что\s+такое\s+(.+?)(?:\?|$|\.)',
        r'поясни\s+(.+?)(?:\?|$|\.)',
        r'расскажи\s+про\s+(.+?)(?:\?|$|\.)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    
    # Если паттерны не сработали, берем последние 2-3 слова
    words = query.split()
    if len(words) > 2:
        return " ".join(words[-3:])
    
    return query

def _extract_topic_from_query(query: str) -> str:
    """Извлекает тему из запроса"""
    patterns = [
        r'найди\s+(?:материал[ы]?|источник[и]?)\s+по\s+(.+?)(?:\?|$|\.)',
        r'материал[ы]?\s+по\s+(.+?)(?:\?|$|\.)',
        r'источник[и]?\s+по\s+(.+?)(?:\?|$|\.)',
        r'книг[и]?\s+по\s+(.+?)(?:\?|$|\.)',
        r'учебник[и]?\s+по\s+(.+?)(?:\?|$|\.)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1).strip()
    
    # Если паттерны не сработали, ищем ключевые слова после "по"
    if 'по' in query.lower():
        parts = query.lower().split('по', 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    return ""

def _extract_timeframe_from_query(query: str) -> str:
    """Извлекает сроки из запроса"""
    patterns = [
        r'на\s+(\d+\s*(?:день|дня|дней|недел[юи]|месяц))',
        r'за\s+(\d+\s*(?:день|дня|дней|недел[юи]|месяц))',
        r'в\s+течение\s+(\d+\s*(?:день|дня|дней|недел[юи]|месяц))'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    
    return ""

def get_help_message() -> str:
    return """🤖 **StudyMate - Помощник по конспектам**

Я помогу вам работать с учебными конспектами:

📚 **Основные возможности:**
• 🧠 Объяснение сложных понятий
• 📚 Поиск учебных материалов  
• 🎯 Учебные советы и методики
• 📝 Улучшение конспектов
• 📅 Создание учебных планов
• 💡 Ответы на вопросы по конспекту

💡 **Примеры запросов:**
• "Объясни что такое дифференциальное исчисление"
• "Найди материалы по квантовой физике"
• "Как лучше учить исторические даты?"
• "Помоги улучшить мой конспект"
• "Создай учебный план на неделю по химии"
• "Какие методы запоминания самые эффективные?"

📁 **Как начать:**
1. Загрузите свой конспект (PDF файл)
2. Задавайте вопросы по вашему материалу
3. Получайте персонализированные объяснения и советы

🚀 **Просто отправьте мне PDF с конспектом и начните общение!**"""

# Экспортируем функции для использования в других модулях
__all__ = [
    'handle_document_upload',
    'handle_user_query',
    'get_help_message'
]

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
from src.agents.quiz_agent import QuizAgent
from src.tools.security import filter_input_query
from src.tools.security import moderate_output_response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
_pending_quiz_count: Dict[int, bool] = {}
_pending_quiz_topic: Dict[int, str] = {}
# Создаем экземпляры агентов
_rag_agent = RAGAgent()
_concept_explainer = ConceptExplainerAgent()
_source_finder = SourceFinderAgent()
_study_advisor = StudyAdvisorAgent()
_quiz_agent = QuizAgent()


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
            try:
                _ = _rag_agent.run(user_id, "инициализация сессии")
            except Exception as e:
                logger.warning(f"Не удалось заранее инициализировать RAG-сессию: {e}")
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
    try:
        logger.info(f"💬 Запрос от студента {user_id}: {query}")
        text_lower = query.lower().strip()

        # Простые команды
        if text_lower in ['/start', '/help', 'помощь', 'help']:
            return get_help_message()

        # Квиз — только детект, вся логика дальше в handle_quiz
        quiz_triggers = [
            'квиз', 'тест', 'сделай квиз',
            'составь квиз', 'сделай тест', 'составь тест',
        ]
        if any(t in text_lower for t in quiz_triggers) or text_lower == '/quiz':
            return await handle_quiz(user_id, query)

        query_type = _analyze_query_type(query)
        logger.info(f"🔍 Тип запроса определен как: {query_type}")

        if query_type in ["study_advice", "notes_improvement", "study_plan", "source_finding", "concept_explanation"]:
            # Эти агенты не отвечают на основе конспекта, они генерируют советы/планы.
            # RAG для них не нужен, поэтому перенаправляем сразу.
            if query_type == "study_advice":
                return await _handle_study_advice(user_id, query)
            elif query_type == "notes_improvement":
                return await _handle_notes_improvement(user_id, query)
            elif query_type == "study_plan":
                return await _handle_study_plan(user_id, query)
            elif query_type == "source_finding":
                return await _handle_source_finding(user_id, query)
            elif query_type == "concept_explanation":
                return await _handle_concept_explanation(user_id, query)

        rag_response = await asyncio.to_thread(_rag_agent.run, user_id, query)

        # print(rag_response)
        if rag_response != "NO_RAG_ANSWER":
            # RAG смог ответить (случай A)
            return rag_response



        # 3. Направляем к соответствующему агенту в зависимости от типа

        else:
            # 4. Если тип не определен, пробуем ConceptExplainer как резервный вариант
            return await _try_concept_explainer_fallback(user_id, query)



    except FileNotFoundError:
        return "⚠️ Сначала загрузите конспект! Используйте команду /start для помощи."

    except Exception as e:
        logger.error(f"❌ Ошибка обработки запроса: {e}")
        return "❌ Произошла ошибка. Попробуйте переформулировать вопрос."


async def _try_rag_response(user_id: int, query: str) -> Dict[str, Any]:
    """
    Пытается найти ответ через RAG систему
    Возвращает dict с флагом успеха и ответом
    """
    try:
        response = await asyncio.to_thread(_rag_agent.run, user_id, query)

        # Анализируем качество ответа RAG
        is_good_response = _evaluate_rag_response(response, query)

        return {
            "success": is_good_response,
            "response": response
        }

    except Exception as e:
        logger.warning(f"RAG не смог обработать запрос: {e}")
        return {
            "success": False,
            "response": ""
        }


def _evaluate_rag_response(response: str, original_query: str) -> bool:
    """
    Оценивает качество ответа RAG
    """
    print(response)
    if not response or len(response.strip()) < 10:
        return False

    # Проверяем признаки неудачного ответа
    negative_indicators = [
        "не удалось найти",
        "нет информации",
        "не могу найти",
        "не знаю",
        "не найдено",
        "у вас нет загруженного",
        "сначала загрузите",
        "не удалось получить ответ",
        "Не нашёл",
        "❌",
        "⚠️"
    ]

    # Проверяем, содержит ли ответ полезную информацию
    response_lower = response.lower()
    has_negative_indicator = any(indicator in response_lower for indicator in negative_indicators)

    # Проверяем релевантность ответа запросу
    query_keywords = _extract_keywords(original_query)
    response_keywords = _extract_keywords(response)

    # Если есть совпадения ключевых слов и нет негативных индикаторов - ответ хороший
    keyword_overlap = len(set(query_keywords) & set(response_keywords))
    has_relevance = keyword_overlap > 0 or len(response) > 50

    return has_relevance and not has_negative_indicator


def _extract_keywords(text: str) -> List[str]:
    """Извлекает ключевые слова из текста"""
    stop_words = {'что', 'как', 'почему', 'где', 'когда', 'объясни', 'найди', 'дай', 'расскажи', 'пожалуйста', 'можно'}
    words = re.findall(r'\b[а-яa-z]{3,}\b', text.lower())
    return [word for word in words if word not in stop_words]


def _analyze_query_type(query: str) -> str:
    """
    Анализирует тип запроса для выбора подходящего агента
    """
    query_lower = query.lower()

    # Концепты и объяснения (высокий приоритет после RAG)
    concept_patterns = [
        r'объясни\s+(?:что\s+такое\s+)?',
        r'что\s+такое\s+',
        r'поясни\s+',
        r'расскажи\s+про\s+',
        r'определи\s+',
        r'в чем смысл',
        r'что значит'
    ]

    # Поиск источников
    source_patterns = [
        r'найди\s+(?:материал[ы]?|источник[и]?)',
        r'материал[ы]?\s+по\s+',
        r'источник[и]?\s+по\s+',
        r'книг[и]?\s+по\s+',
        r'учебник[и]?\s+по\s+',
        r'литератур[ау]?\s+по\s+',
        r'где найти',
        r'посоветуй книг'
    ]

    # Учебные советы
    advice_patterns = [
        r'как\s+(?:лучше|эффективно)\s+(?:учит|изуча|запомина)',
        r'совет[ы]?\s+по\s+(?:учёбе|изучен)',
        r'метод[ы]?\s+обучен',
        r'как\s+запоминать',
        r'техник[и]?\s+запоминан',
        r'учебн[ые]?\s+совет[ы]?',
        r'как\s+готовиться'
    ]

    # Улучшение конспектов
    notes_patterns = [
        r'улучши\s+',
        r'как\s+вести\s+конспект',
        r'совет[ы]?\s+по\s+конспект',
        r'структур[ау]?\s+заметок',
        r'оформи\s+конспект',
        r'метод[ы]?\s+конспектирован'
    ]

    # Учебные планы
    plan_patterns = [
        r'план\s+(?:изучен|обучен)',
        r'расписание\s+занятий',
        r'график\s+изучен',
        r'распредели\s+по\s+дням',
        r'составь\s+план',
        r'как\s+спланировать'
    ]

    # Проверяем паттерны в порядке приоритета
    if any(re.search(pattern, query_lower) for pattern in concept_patterns):
        return "concept_explanation"
    elif any(re.search(pattern, query_lower) for pattern in source_patterns):
        return "source_finding"
    elif any(re.search(pattern, query_lower) for pattern in advice_patterns):
        return "study_advice"
    elif any(re.search(pattern, query_lower) for pattern in notes_patterns):
        return "notes_improvement"
    elif any(re.search(pattern, query_lower) for pattern in plan_patterns):
        return "study_plan"
    else:
        return "general"


async def _try_concept_explainer_fallback(user_id: int, query: str) -> str:
    """
    Резервный вариант - пытаемся объяснить запрос как концепт
    """
    try:
        # Извлекаем возможный концепт из запроса
        concept = _extract_possible_concept(query)

        if concept:
            logger.info(f"🔄 Использую ConceptExplainer для концепта: {concept}")
            explanation_result = await asyncio.to_thread(
                _concept_explainer.explain_concept,
                concept,
                f"Запрос пользователя: {query}"
            )

            if explanation_result and "explanation" in explanation_result:
                response = f"🧠 **Объяснение: {concept}**\n\n"
                response += explanation_result["explanation"]

                if "key_points" in explanation_result:
                    response += f"\n\n🔑 **Ключевые моменты:**\n"
                    for point in explanation_result["key_points"][:3]:
                        response += f"• {point}\n"

                return moderate_output_response(response)

        # Если концепт не извлекли или объяснение не удалось
        return "🤔 Не удалось найти информацию в вашем конспекте. Попробуйте переформулировать вопрос или уточнить, что именно вас интересует."

    except Exception as e:
        logger.error(f"❌ Ошибка в резервном ConceptExplainer: {e}")
        return "❌ Не удалось обработать ваш запрос. Попробуйте задать вопрос по-другому."


def _extract_possible_concept(query: str) -> str:
    """
    Пытается извлечь концепт из общего запроса
    """
    question_words = {'что', 'как', 'почему', 'где', 'когда', 'зачем', 'какой', 'какая', 'какое', 'какие'}
    words = query.lower().split()

    # Ищем существительные и важные термины
    content_words = [word for word in words if word not in question_words and len(word) > 3]

    if len(content_words) >= 2:
        return " ".join(content_words[-2:])
    elif content_words:
        return content_words[-1]
    else:
        return " ".join(words[1:]) if len(words) > 1 else query


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

            return moderate_output_response(response)
        else:
            # Если агент не смог объяснить, используем RAG
            return await asyncio.to_thread(_rag_agent.run, user_id, f"Объясни понятие: {concept}")

    except Exception as e:
        logger.error(f"❌ Ошибка объяснения: {e}")
        return f"❌ Не удалось объяснить понятие. Попробуйте задать вопрос по-другому."


async def _get_context_from_rag(user_id: int, query: str) -> str:
    """Получает контекст из RAG, возвращает пустую строку, если RAG не нашел ответ."""
    try:
        # ⚠️ Здесь важно, чтобы _rag_agent.run возвращал чистый сигнал "NO_RAG_ANSWER" при неудаче
        context_response = await asyncio.to_thread(_rag_agent.run, user_id, query)

        # Проверка на сигнал неудачи
        if context_response == "NO_RAG_ANSWER" or "нет информации" in context_response.lower():
            return ""

        # Если RAG что-то ответил, возвращаем это как контекст (ограничиваем длину)
        if len(context_response) > 1000:
            return context_response[:1000] + "..."

        return context_response

    except Exception as e:
        logger.warning(f"Не удалось получить контекст из RAG: {e}")
        return ""


async def _handle_source_finding(user_id: int, query: str) -> str:
    try:
        topic = _extract_topic_from_query(query)

        if not topic:
            return "❌ Не смог определить тему для поиска. Попробуйте: 'Найди материалы по [теме]'"

        # Получаем контекст, чтобы LLM мог дать персонализированные рекомендации
        context = await _get_context_from_rag(user_id, topic)

        sources_result = await asyncio.to_thread(
            _source_finder.find_sources,
            topic,
            context
        )

        if sources_result and "sources" in sources_result:
            response = f"📚 **Материалы по теме: {topic}**\n\n"

            # 💡 ФИКС: Используем словарь, который уже сгруппирован агентом
            sources_by_type = sources_result["sources"]

            for normalized_type, sources in sources_by_type.items():
                if not sources:
                    continue

                # Используем тип с эмодзи из первого элемента (он есть, если агент корректно отработал)
                source_type_display = sources[0].get('type_with_emoji', normalized_type.upper())

                response += f"**{source_type_display}:**\n"

                # Выводим до 3 источников каждого типа
                for source in sources[:3]:
                    level = source.get('level', 'N/A')
                    language = source.get('language', 'N/A')

                    response += f"• **{source['name']}** ({level.capitalize()})"

                    if source.get('description'):
                        response += f"\n  — *{source['description']}*"

                    response += f" [{language.capitalize()}]"

                    response += "\n"
                response += "\n"

            if "study_path" in sources_result:
                response += f"🎯 **Рекомендуемый порядок изучения:**\n"
                for stage in sources_result["study_path"][:3]:
                    response += f"• {stage}\n"

            if not context:
                response += f"\n---\n*ℹ️ Эти рекомендации общие. Для персонализированных материалов загрузите свой конспект.*"

            return moderate_output_response(response)
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

                return moderate_output_response(response)

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

            return moderate_output_response(response)
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

            return moderate_output_response(response)
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

            return moderate_output_response(response)
        else:
            return "❌ Не удалось создать учебный план."

    except Exception as e:
        logger.error(f"❌ Ошибка создания плана: {e}")
        return "❌ Произошла ошибка при создании учебного плана."

async def handle_quiz(user_id: int, query: str) -> str:
    """
    Вся логика работы с квизом (диалог + генерация).
    Вызывается из handle_user_query.
    """
    return await _handle_quiz(user_id, query)

async def _handle_quiz(user_id: int, query: str) -> str:
    try:
        text_lower = query.lower().strip()

        # 1) Первый шаг: пользователь только что написал "квиз/тест"
        if user_id not in _pending_quiz_topic and user_id not in _pending_quiz_count:
            _pending_quiz_topic[user_id] = ""
            _pending_quiz_count[user_id] = False
            return ('По какой теме сделать квиз? Напишите тему из конспекта '
                    'или слово "весь" для квиза по всему конспекту.')

        # 2) Ещё нет темы — текущий ввод считаем темой
        if _pending_quiz_topic.get(user_id, "") == "":
            topic_text = text_lower.strip()
            if not topic_text:
                return ('Пожалуйста, укажите тему или слово "весь" '
                        'для квиза по всему конспекту.')
            _pending_quiz_topic[user_id] = topic_text
            _pending_quiz_count[user_id] = True
            return "На сколько вопросов сделать квиз? Напишите число от 1 до 10."

        # 3) Ждём число вопросов
        if _pending_quiz_count.get(user_id) and text_lower.isdigit():
            n = int(text_lower)
            if not (1 <= n <= 10):
                return "Пожалуйста, введите число от 1 до 10."

            _pending_quiz_count[user_id] = False
            topic = _pending_quiz_topic.get(user_id, "весь")

            # 3.1 Контекст по теме или по всему конспекту
            if topic and topic != "весь":
                context = await _get_context_from_notes(user_id, topic)
            else:
                context = await asyncio.to_thread(_rag_agent.get_note_text, user_id)

            print("QUIZ CONTEXT LEN:", len(context))
            if not context:
                _pending_quiz_topic.pop(user_id, None)
                return ("❌ Не удалось найти текст конспекта по этой теме. "
                        "Попробуйте другую формулировку или слово \"весь\".")

            # 3.2 Генерация квиза
            quiz_data = await asyncio.to_thread(
                _quiz_agent.generate_quiz,
                context,
                n,
                topic,          # если добавлял topic в сигнатуру
            )
            questions = quiz_data.get("questions", [])
            if not questions:
                _pending_quiz_topic.pop(user_id, None)
                return "❌ Не удалось сгенерировать quiz. Попробуйте ещё раз."

            # 3.3 Формирование HTML‑ответа со спойлерами
            response = f"📝 <i>Quiz по вашему конспекту</i> \n({len(questions)} вопросов)\n\n"
            for i, q in enumerate(questions, 1):
                question = q.get("question", "Вопрос")
                options = q.get("options", [])
                correct = q.get("correct_answer", "")
                explanation = q.get("explanation", "")

                response += f"<b>{i}. {question}</b>\n\n"
                for opt in options:
                    response += f"• {opt}\n"
                response += f"\nОтвет: <tg-spoiler>{correct}</tg-spoiler>\n"
                if explanation:
                    response += f"Объяснение: <tg-spoiler>{explanation}</tg-spoiler>\n"
                response += "\n──────────────\n\n"

            _pending_quiz_topic.pop(user_id, None)
            _pending_quiz_count.pop(user_id, None)
            return response

        # 4) Если ждём число, а пришла не цифра
        if _pending_quiz_count.get(user_id) and not text_lower.isdigit():
            return "Пожалуйста, введите число от 1 до 10."

        # 5) Фолбэк: сброс состояния
        _pending_quiz_topic.pop(user_id, None)
        _pending_quiz_count.pop(user_id, None)
        return 'Давайте начнём квиз сначала. Напишите "квиз", чтобы запустить тест.'

    except Exception as e:
        logger.error(f"❌ Ошибка в _handle_quiz: {e}")
        return "❌ Произошла ошибка при создании quiz."

def get_retrieved_context(self, topic: str, k: int = 4) -> str:
    """
    Возвращает ЧИСТЫЙ извлеченный текст (чанки), ИГНОРИРУЯ ПАМЯТЬ и LLM.
    Используется только для предоставления контекста другим агентам.
    """
    # 1. Используем чистый ретривер (из RAGLoader)
    docs = self.qa_chain.retriever.get_relevant_documents(topic)  # self.qa_chain.retriever - это ваш retriever

    # 2. Объединяем в одну строку
    context = "\n---\n".join([doc.page_content for doc in docs])

    # 3. Ограничиваем длину (для Concept Explainer)
    if len(context) > 2000:
        return context[:2000] + " [Контекст обрезан для передачи агенту]"

    return context

async def _get_context_from_notes(user_id: int, query: str, max_chars: int = 4000) -> str:
    """Получает релевантный контекст из конспектов пользователя по теме query."""
    try:
        session = _rag_agent._get_or_create_session(user_id)
        retriever = getattr(session, "retriever", None)
        if retriever is None:
            logger.warning("get_context_from_notes: у session нет поля retriever")
            return ""

        docs = await asyncio.to_thread(retriever.get_relevant_documents, query)
        if not docs:
            logger.info(f"get_context_from_notes: retriever вернул 0 документов по теме '{query}'")
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

        context = "\n\n".join(chunks)
        logger.info(f"get_context_from_notes: длина контекста по теме '{query}' = {len(context)}")
        return context

    except Exception as e:
        logger.warning(f"Не удалось получить контекст по теме '{query}': {e}")
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

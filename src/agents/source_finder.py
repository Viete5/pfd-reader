import logging
import re
from typing import List, Dict, Any
from langchain_gigachat.chat_models import GigaChat
from src.services.get_token import get_token
from src.config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class SourceFinderAgent:
    """
    Расширенный агент для поиска релевантных учебных источников
    с интеллектуальной базой знаний и категоризацией тем
    """

    def __init__(self):
        self.llm = self._initialize_llm()
        self.knowledge_bases = self._initialize_knowledge_bases()
        self.source_types = self._initialize_source_types()

    def _initialize_llm(self):
        """Инициализация GigaChat"""
        token = get_token()
        return GigaChat(
            temperature=LLM_TEMPERATURE,
            verify_ssl_certs=False,
            access_token=token
        )

    def _initialize_knowledge_bases(self) -> Dict[str, Dict[str, List[str]]]:
        """Расширенная база знаний с источниками по различным дисциплинам"""
        return {
            "physics": {
                "textbooks": [
                    "Фейнмановские лекции по физике - Р. Фейнман",
                    "Берклеевский курс физики - Э. Вихман, Р. Пурселл",
                    "Курс общей физики - И.В. Савельев (3 тома)",
                    "Общий курс физики - Д.В. Сивухин (5 томов)",
                    "Теоретическая физика - Л.Д. Ландау, Е.М. Лифшиц (10 томов)",
                    "Физика для всех - Л. Купер",
                    "Элементарный учебник физики - Г.С. Ландсберг (3 тома)",
                    "University Physics with Modern Physics - Young and Freedman"
                ],
                "online_courses": [
                    "Coursera - Физика: механика, колебания и волны (МФТИ)",
                    "Stepik - Основы физики",
                    "Открытое образование - Механика (МГУ)",
                    "MIT OpenCourseWare - Physics I: Classical Mechanics",
                    "Khan Academy - Physics",
                    "edX - Introduction to Mechanics (MIT)",
                    "Лекториум - Курс общей физики"
                ],
                "video_lectures": [
                    "Лекции по физике от МФТИ (YouTube канал)",
                    "Физика от Образовача (Telegram канал)",
                    "Susskind's Theoretical Minimum (Stanford)",
                    "Walter Lewin Lectures - MIT (YouTube)",
                    "Лекции по квантовой механике - В.В. Белоусов"
                ],
                "interactive_platforms": [
                    "PhET Interactive Simulations (University of Colorado)",
                    "Wolfram Physics Project",
                    "Brilliant.org - Physics courses",
                    "The Physics Classroom (interactive tutorials)"
                ],
                "reference_books": [
                    "Физический энциклопедический словарь",
                    "Справочник по физике - Яворский, Детлаф",
                    "Tables of Physical and Chemical Constants"
                ]
            },

            "mathematics": {
                "textbooks": [
                    "Высшая математика - В.Е. Шнейдер, А.И. Слуцкий, А.С. Шумов",
                    "Курс высшей математики - В.И. Смирнов (5 томов)",
                    "Математический анализ - Г.М. Фихтенгольц (3 тома)",
                    "Дифференциальные уравнения - Л.С. Понтрягин",
                    "Линейная алгебра и аналитическая геометрия - А.В. Ильин, Г.Д. Ким",
                    "Теория функций комплексной переменной - А.Г. Свешников, А.Н. Тихонов",
                    "Курс дифференциального и интегрального исчисления - Г.М. Фихтенгольц",
                    "Advanced Engineering Mathematics - Erwin Kreyszig"
                ],
                "online_courses": [
                    "Coursera - Математика для физиков (МФТИ)",
                    "Stepik - Линейная алгебра (МФТИ)",
                    "MIT OpenCourseWare - Single Variable Calculus",
                    "Khan Academy - Calculus",
                    "edX - Introduction to Differential Equations (MIT)",
                    "Лекториум - Высшая математика",
                    "Универсариум - Математический анализ"
                ],
                "problem_books": [
                    "Сборник задач по высшей математике - Л.А. Кузнецов",
                    "Задачи и упражнения по математическому анализу - Б.П. Демидович",
                    "Сборник задач по математике для ВТУЗов - Ефимов, Демидович",
                    "3000 конкурсных задач по математике - Куланин, Норин"
                ],
                "video_lectures": [
                    "Лекции по математике от мехмата МГУ",
                    "Математика - это просто (YouTube канал)",
                    "3Blue1Brown - Essence of Calculus/Linear Algebra",
                    "Лекции по матанализу - В.В. Прасолов"
                ]
            },

            "programming": {
                "textbooks": [
                    "Современный учебник JavaScript - learn.javascript.ru",
                    "Python Crash Course - Eric Matthes",
                    "Clean Code: A Handbook of Agile Software Craftsmanship - Robert C. Martin",
                    "Грокаем алгоритмы - Адитья Бхаргава",
                    "Структура и интерпретация компьютерных программ - Харольд Абельсон, Джеральд Сассман",
                    "Design Patterns: Elements of Reusable Object-Oriented Software - GoF",
                    "Introduction to Algorithms - Cormen, Leiserson, Rivest, Stein",
                    "The C Programming Language - Kernighan & Ritchie"
                ],
                "online_courses": [
                    "Coursera - Python for Everybody (University of Michigan)",
                    "Stepik - Программирование на Python (МФТИ)",
                    "freeCodeCamp - Full Stack Development",
                    "Codecademy - Interactive coding courses",
                    "Harvard CS50 - Introduction to Computer Science",
                    "Hexlet - JavaScript и веб-разработка",
                    "Яндекс.Практикум - Веб-разработка"
                ],
                "practice_platforms": [
                    "LeetCode - алгоритмические задачи",
                    "HackerRank - coding challenges",
                    "Codewars - katas for skill improvement",
                    "Exercism - practice with mentors",
                    "Project Euler - mathematical programming",
                    "Codeforces - competitive programming"
                ],
                "video_tutorials": [
                    "Уроки Python от Stepik (YouTube)",
                    "JavaScript Mastery (YouTube)",
                    "Traversy Media - Web Development Tutorials",
                    "The Net Ninja - Programming Tutorials"
                ],
                "documentation": [
                    "MDN Web Docs - web technologies reference",
                    "Python Official Documentation",
                    "React Documentation",
                    "Vue.js Guide"
                ]
            },

            "chemistry": {
                "textbooks": [
                    "Общая химия - Н.Л. Глинка",
                    "Курс физической химии - Я.И. Герасимов",
                    "Органическая химия - А.П. Лузин, С.Э. Зурабян",
                    "Неорганическая химия - Ю.Д. Третьяков",
                    "Аналитическая химия - А.А. Ищенко",
                    "Physical Chemistry - P. Atkins",
                    "Organic Chemistry - J. McMurry"
                ],
                "online_courses": [
                    "Coursera - Химия (МГУ)",
                    "Stepik - Общая химия",
                    "Khan Academy - Chemistry",
                    "Лекториум - Органическая химия",
                    "Открытое образование - Неорганическая химия"
                ],
                "video_lectures": [
                    "Лекции по химии от химфака МГУ",
                    "Chemistry from University of Nottingham (YouTube)",
                    "Organic Chemistry Tutor - YouTube channel"
                ],
                "virtual_labs": [
                    "ChemCollective Virtual Labs",
                    "PhET Chemistry Simulations",
                    "Labster Chemistry Labs"
                ]
            },

            "biology": {
                "textbooks": [
                    "Биология - В.Н. Ярыгин (2 тома)",
                    "Общая биология - Д.К. Беляев",
                    "Молекулярная биология клетки - Б. Альбертс",
                    "Генетика - М.Е. Лобашев",
                    "Биохимия - Л. Страйер",
                    "Campbell Biology - комплексный учебник"
                ],
                "online_courses": [
                    "Coursera - Генетика (МГУ)",
                    "Stepik - Молекулярная биология",
                    "Khan Academy - Biology",
                    "edX - Introduction to Biology (MIT)",
                    "Лекториум - Общая биология"
                ],
                "video_resources": [
                    "Лекции по биологии от биофака МГУ",
                    "Bozeman Science - Biology (YouTube)",
                    "Crash Course Biology (YouTube)",
                    "Amoeba Sisters - биологические концепты (YouTube)"
                ],
                "databases": [
                    "NCBI - National Center for Biotechnology Information",
                    "UniProt - Protein Database",
                    "PDB - Protein Data Bank"
                ]
            },

            "engineering": {
                "textbooks": [
                    "Теоретическая механика - Н.В. Бутенин",
                    "Сопротивление материалов - А.В. Александров",
                    "Детали машин - М.Н. Иванов",
                    "Теория механизмов и машин - И.И. Артоболевский",
                    "Электротехника - Л.А. Бессонов",
                    "Materials Science and Engineering: An Introduction - Callister"
                ],
                "online_courses": [
                    "Coursera - Основы инженерии",
                    "edX - Mechanical Engineering (MIT)",
                    "MIT OpenCourseWare - Engineering",
                    "Лекториум - Теоретическая механика"
                ],
                "software_tools": [
                    "MATLAB - numerical computing",
                    "AutoCAD - computer-aided design",
                    "SolidWorks - 3D CAD design",
                    "ANSYS - engineering simulation",
                    "COMSOL Multiphysics - simulation software"
                ],
                "standards": [
                    "ГОСТы по машиностроению",
                    "ISO Standards (International Organization for Standardization)",
                    "IEEE Standards (Institute of Electrical and Electronics Engineers)"
                ]
            },

            "economics": {
                "textbooks": [
                    "Экономика - С.Г. Капканщиков",
                    "Микроэкономика - Р. Пиндайк, Д. Рубинфельд",
                    "Макроэкономика - Н.Г. Мэнкью",
                    "Эконометрика - Дж. М. Вулдридж",
                    "Principles of Economics - N. Gregory Mankiw"
                ],
                "online_courses": [
                    "Coursera - Экономика для неэкономистов (ВШЭ)",
                    "edX - Microeconomics (MIT)",
                    "Khan Academy - Economics",
                    "Stepik - Основы экономики"
                ],
                "data_sources": [
                    "World Bank Open Data",
                    "IMF Data - International Monetary Fund",
                    "Federal Reserve Economic Data (FRED)",
                    "Росстат - Официальная статистика России"
                ],
                "research_journals": [
                    "American Economic Review",
                    "Journal of Economic Perspectives",
                    "Вопросы экономики (российский журнал)"
                ]
            },

            "history": {
                "textbooks": [
                    "История России - А.С. Орлов, В.А. Георгиев",
                    "Всемирная история - О.В. Волобуев",
                    "История Древнего мира - В.И. Кузищин",
                    "История Средних веков - С.П. Карпов",
                    "Новая история - Ю.В. Кудрина"
                ],
                "online_courses": [
                    "Coursera - История России (ВШЭ)",
                    "Arzamas Academy - История культуры",
                    "ПостНаука - Исторические курсы",
                    "Лекториум - Всемирная история"
                ],
                "primary_sources": [
                    "Русская правда",
                    "Судебники Ивана III и Ивана IV",
                    "Соборное уложение 1649 года",
                    "Конституционные акты Российской империи"
                ],
                "digital_archives": [
                    "РГАДА - Российский государственный архив древних актов",
                    "Project Gutenberg - Historical Texts",
                    "Internet Archive - Historical collections"
                ]
            },

            "languages": {
                "textbooks": [
                    "English Grammar in Use - Raymond Murphy",
                    "Практическая грамматика английского языка - К.Н. Качалова",
                    "Assimil language courses",
                    "Practice Makes Perfect series"
                ],
                "online_platforms": [
                    "Duolingo - gamified language learning",
                    "Busuu - language community",
                    "Memrise - vocabulary building",
                    "Lingualeo - English for Russians",
                    "BBC Learning English"
                ],
                "practice_tools": [
                    "Anki - spaced repetition flashcards",
                    "HelloTalk - language exchange app",
                    "Tandem - practice with native speakers",
                    "Forvo - pronunciation guide"
                ],
                "certification": [
                    "Cambridge English exams preparation",
                    "TOEFL official practice materials",
                    "IELTS test preparation resources",
                    "TestDaF for German language"
                ]
            },

            "computer_science": {
                "textbooks": [
                    "Computer Systems: A Programmer's Perspective - R. Bryant, D. O'Hallaron",
                    "Operating System Concepts - Silberschatz, Galvin, Gagne",
                    "Computer Networks - A. Tanenbaum",
                    "Database System Concepts - Silberschatz, Korth, Sudarshan",
                    "Artificial Intelligence: A Modern Approach - Russell, Norvig"
                ],
                "online_courses": [
                    "CS50 - Harvard University Introduction to Computer Science",
                    "Coursera - Computer Science specialization",
                    "edX - CS Fundamentals (MIT)",
                    "Stanford Online - CS courses",
                    "MIT OpenCourseWare - Computer Science"
                ],
                "programming_competitions": [
                    "ACM ICPC - International Collegiate Programming Contest",
                    "Google Code Jam",
                    "Facebook Hacker Cup",
                    "Russian Code Cup"
                ],
                "research_journals": [
                    "Communications of the ACM",
                    "IEEE Transactions on Software Engineering",
                    "Journal of Machine Learning Research"
                ]
            },

            "medicine": {
                "textbooks": [
                    "Анатомия человека - М.Р. Сапин",
                    "Физиология человека - Р. Шмидт, Г. Тевс",
                    "Патологическая анатомия - А.И. Струков, В.В. Серов",
                    "Внутренние болезни - В.И. Маколкин, С.И. Овчаренко"
                ],
                "online_courses": [
                    "Coursera - Основы медицины",
                    "edX - Human Anatomy",
                    "MedlinePlus - медицинская информация"
                ]
            },

            "psychology": {
                "textbooks": [
                    "Общая психология - А.Г. Маклаков",
                    "Психология личности - Л. Хьелл, Д. Зиглер",
                    "Социальная психология - Д. Майерс"
                ],
                "online_courses": [
                    "Coursera - Introduction to Psychology (Yale)",
                    "edX - Psychology courses"
                ]
            }
        }

    def _initialize_source_types(self) -> Dict[str, str]:
        """Типы источников с эмодзи для красивого отображения"""
        return {
            "textbook": "📚 Учебник/Книга",
            "online_course": "🎓 Онлайн-курс",
            "video_lecture": "🎥 Видео-лекция",
            "practice_platform": "💻 Практическая платформа",
            "interactive_sim": "🔬 Интерактивная симуляция",
            "reference_book": "📖 Справочник",
            "research_paper": "📄 Научная статья",
            "documentation": "📋 Документация",
            "community_forum": "👥 Сообщество/Форум",
            "podcast": "🎧 Подкаст/Аудио",
            "newsletter": "📰 Рассылка/Блог",
            "problem_book": "📝 Сборник задач",
            "virtual_lab": "🧪 Виртуальная лаборатория",
            "database": "🗄️ База данных",
            "software_tool": "🛠️ Программное обеспечение",
            "standard": "📏 Стандарт/ГОСТ",
            "data_source": "📊 Источник данных",
            "primary_source": "📜 Первоисточник",
            "digital_archive": "🗃️ Цифровой архив",
            "certification": "🏆 Подготовка к сертификации"
        }

    def _categorize_topic(self, topic: str) -> str:
        """
        Интеллектуальное определение категории темы по ключевым словам
        """
        topic_lower = topic.lower()

        category_keywords = {
            "physics": [
                'физик', 'механ', 'электр', 'квант', 'термодинам', 'оптик',
                'акустик', 'ядерн', 'релятив', 'гравитац', 'поле', 'волн',
                'частиц', 'атом', 'молекул', 'планк', 'ньютон', 'эйнштейн',
                'магнит', 'заряд', 'энерг', 'масс', 'сил'
            ],
            "mathematics": [
                'математ', 'алгебр', 'геометр', 'анализ', 'дифференц', 'интеграл',
                'уравнен', 'теория вероятност', 'статистик', 'тфкп', 'линейн',
                'тополог', 'численн', 'логик', 'дискретн', 'комбинатор',
                'исчислен', 'функц', 'предел', 'производн'
            ],
            "programming": [
                'программир', 'код', 'алгоритм', 'структур', 'баз', 'python',
                'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'веб', 'frontend', 'backend', 'api', 'фреймворк', 'библиотек',
                'разработк', 'тестирован', 'дебаг', 'компиляц'
            ],
            "chemistry": [
                'хими', 'орган', 'неорган', 'аналит', 'физич', 'биохим',
                'молекул', 'атом', 'реакц', 'соединен', 'периодическ', 'элемент',
                'веществ', 'связ', 'валентност', 'кислот', 'основан'
            ],
            "biology": [
                'биолог', 'генет', 'клетк', 'эволюц', 'экологи', 'анатом',
                'физиолог', 'микробиолог', 'биохим', 'молекул', 'днк', 'рнк',
                'белок', 'фермент', 'орган', 'ткань', 'вид', 'популяц'
            ],
            "engineering": [
                'инженер', 'механик', 'конструкц', 'черчен', 'сопротивлен',
                'материал', 'детал', 'механизм', 'электротех', 'схем', 'проект',
                'расчет', 'чертеж', 'техническ', 'производств'
            ],
            "economics": [
                'экономик', 'финанс', 'бухгалтер', 'менеджмент', 'маркетинг',
                'бизнес', 'предприниматель', 'акци', 'облигац', 'рынок',
                'инвестиц', 'кредит', 'налог', 'бюджет', 'валов'
            ],
            "history": [
                'истори', 'древн', 'средневеков', 'новое время', 'современн',
                'археолог', 'цивилизац', 'культур', 'общество', 'государство',
                'войн', 'революц', 'импери', 'династи', 'хронологи'
            ],
            "languages": [
                'английск', 'немецк', 'французск', 'испанск', 'китайск',
                'японск', 'корейск', 'итальянск', 'португальск', 'грамматик',
                'лексик', 'произношен', 'перевод', 'язык', 'лингвист'
            ],
            "computer_science": [
                'компьютерн', 'информатик', 'ос ', 'сеть', 'баз', 'искусственный интеллект',
                'машинное обучение', 'нейрон', 'big data', 'кибербезопасност',
                'компьютерн', 'архитектур', 'протокол', 'шифрован'
            ]
        }

        for category, keywords in category_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return category

        # Проверяем по более общим шаблонам
        if re.search(r'(?:учеб|материал|книг|курс)', topic_lower):
            return "general"

        return "general"  # Общая категория по умолчанию

    def find_sources(self, topic: str, context: str = "") -> Dict[str, Any]:
        """
        Находит источники для конкретной темы с расширенным поиском

        Args:
            topic: Тема для поиска источников
            context: Контекст из конспекта студента

        Returns:
            Словарь с источниками, путем изучения и рекомендациями
        """
        try:
            # Определяем категорию темы
            category = self._categorize_topic(topic)
            logger.info(f"Тема '{topic}' определена как категория '{category}'")

            # Получаем источники от LLM с учетом категории
            llm_sources = self._get_sources_from_llm(topic, category, context)

            # Получаем источники из базы знаний
            base_sources = self._get_base_sources(category, topic)

            # Объединяем и структурируем источники
            all_sources = base_sources + llm_sources

            # Группируем источники по типам
            structured_sources = self._structure_sources_by_type(all_sources)

            # Создаем путь изучения
            study_path = self._create_study_path(topic, category, structured_sources)

            # Получаем дополнительные рекомендации
            recommendations = self._get_recommendations(category, topic)

            return {
                "topic": topic,
                "category": category,
                "sources": structured_sources,
                "study_path": study_path,
                "recommendations": recommendations,
                "total_sources": len(all_sources)
            }

        except Exception as e:
            logger.error(f"Ошибка при поиске источников для темы '{topic}': {e}")
            return self._get_fallback_sources(topic)

    def _get_sources_from_llm(self, topic: str, category: str, context: str) -> List[Dict[str, str]]:
        """Получает источники от LLM"""
        prompt = f"""
        Студент изучает тему: "{topic}"
        Категория: {category}
        {f"Контекст из конспекта: {context[:500]}" if context else ""}

        Предложи 5-7 лучших учебных ресурсов, учитывая:
        1. Разнообразие форматов (книги, курсы, видео, практика)
        2. Уровень сложности (укажи: начальный/средний/продвинутый)
        3. Доступность (бесплатные/платные, русский/английский)
        4. Практическую применимость

        Для каждого ресурса укажи:
        - Название
        - Тип (учебник, онлайн-курс, видео-лекция и т.д.)
        - Уровень сложности
        - Краткое описание (1-2 предложения)
        - Где найти/ссылка

        Формат ответа:
        НАЗВАНИЕ: [название]
        ТИП: [тип]
        УРОВЕНЬ: [уровень]
        ОПИСАНИЕ: [описание]
        ---
        """

        try:
            response = self.llm.invoke(prompt)
            return self._parse_llm_response(response.content)
        except Exception as e:
            logger.error(f"Ошибка при получении источников от LLM: {e}")
            return []

    def _get_base_sources(self, category: str, topic: str) -> List[Dict[str, str]]:
        """Получает источники из базы знаний для категории"""
        sources = []

        if category in self.knowledge_bases:
            category_data = self.knowledge_bases[category]

            # Берем по 1-2 источника каждого типа
            for source_type, source_list in category_data.items():
                for i, source_name in enumerate(source_list[:2]):
                    source = {
                        'name': source_name,
                        'type': source_type,
                        'level': self._determine_level(source_name, topic),
                        'description': self._generate_description(source_name, category, source_type),
                        'link': self._generate_search_link(source_name, topic),
                        'language': self._determine_language(source_name),
                        'from_knowledge_base': True  # Флаг, что источник из базы знаний
                    }
                    sources.append(source)

        return sources

    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """Парсит ответ от LLM"""
        sources = []
        current_source = {}

        for line in response.split('\n'):
            line = line.strip()

            if line.startswith('НАЗВАНИЕ:'):
                if current_source:
                    sources.append(current_source)
                current_source = {'name': line.replace('НАЗВАНИЕ:', '').strip()}
            elif line.startswith('ТИП:') and current_source:
                current_source['type'] = line.replace('ТИП:', '').strip()
            elif line.startswith('УРОВЕНЬ:') and current_source:
                current_source['level'] = line.replace('УРОВЕНЬ:', '').strip()
            elif line.startswith('ОПИСАНИЕ:') and current_source:
                current_source['description'] = line.replace('ОПИСАНИЕ:', '').strip()
            elif line == '---' and current_source:
                sources.append(current_source)
                current_source = {}

        if current_source and 'name' in current_source:
            sources.append(current_source)

        return sources

    def _structure_sources_by_type(self, sources: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Структурирует источники по типам"""
        structured = {}

        for source in sources:
            source_type = source.get('type', 'other')

            # Приводим тип к стандартному виду
            normalized_type = self._normalize_source_type(source_type)

            if normalized_type not in structured:
                structured[normalized_type] = []

            # Добавляем эмодзи для типа источника
            source['type_with_emoji'] = self.source_types.get(normalized_type, normalized_type)
            structured[normalized_type].append(source)

        return structured

    def _create_study_path(self, topic: str, category: str,
                           structured_sources: Dict[str, List[Dict[str, str]]]) -> List[str]:
        """Создает рекомендуемый путь изучения"""
        study_path = []

        # Определяем порядок изучения по типам источников
        type_order = ['textbook', 'online_course', 'video_lecture', 'practice_platform']

        for source_type in type_order:
            if source_type in structured_sources:
                sources = structured_sources[source_type]
                if sources:
                    # Берем первый источник этого типа
                    first_source = sources[0]
                    study_path.append(
                        f"Начните с {source_type.replace('_', ' ')}: '{first_source['name']}'"
                    )

        # Добавляем общие рекомендации
        study_path.append("Регулярно практикуйтесь на задачах и упражнениях")
        study_path.append("Объясняйте материал своими словами для лучшего понимания")
        study_path.append("Создавайте собственные конспекты и шпаргалки")

        return study_path

    def _get_recommendations(self, category: str, topic: str) -> Dict[str, Any]:
        """Генерирует дополнительные рекомендации"""
        recommendations = {
            "general": [
                "Изучайте материал регулярно, небольшими порциями",
                "Делайте перерывы каждые 45-50 минут",
                "Сочетайте теорию с практикой"
            ],
            "category_specific": []
        }

        # Рекомендации по категориям
        category_recommendations = {
            "physics": ["Решайте много задач", "Используйте физические симуляторы"],
            "mathematics": ["Доказывайте теоремы самостоятельно", "Решайте задачи разного уровня"],
            "programming": ["Пишите код ежедневно", "Участвуйте в open-source проектах"],
            "chemistry": ["Проводите виртуальные эксперименты", "Изучайте химические модели"],
            "biology": ["Используйте анатомические атласы", "Изучайте биологические процессы наглядно"]
        }

        if category in category_recommendations:
            recommendations["category_specific"] = category_recommendations[category]

        return recommendations

    def _determine_level(self, source_name: str, topic: str) -> str:
        """Определяет уровень сложности источника"""
        # Простая эвристика по названию
        lower_name = source_name.lower()

        if any(word in lower_name for word in ['основ', 'введение', 'начальный', 'базов']):
            return "начальный"
        elif any(word in lower_name for word in ['продвинут', 'advanced', 'углублен']):
            return "продвинутый"
        else:
            return "средний"

    def _generate_description(self, source_name: str, category: str, source_type: str) -> str:
        """Генерирует описание для источника"""
        descriptions = {
            "textbook": f"Классический учебник по {category}",
            "online_course": f"Современный онлайн-курс по {category}",
            "video_lecture": f"Видео-лекции по {category}",
            "practice_platform": f"Платформа для практики по {category}"
        }

        return descriptions.get(source_type, f"Ресурс по {category}")


    def _determine_language(self, source_name: str) -> str:
        """Определяет язык источника"""
        # Простая эвристика
        if any(word in source_name.lower() for word in ['english', 'англ', 'eng']):
            return "английский"
        elif any(word in source_name.lower() for word in ['русск', 'russian']):
            return "русский"
        else:
            return "смешанный"

    def _normalize_source_type(self, source_type: str) -> str:
        """Нормализует тип источника к стандартному виду"""
        type_mapping = {
            'учебник': 'textbook',
            'книга': 'textbook',
            'курс': 'online_course',
            'онлайн-курс': 'online_course',
            'видео': 'video_lecture',
            'лекция': 'video_lecture',
            'платформа': 'practice_platform',
            'практика': 'practice_platform',
            'задачи': 'problem_book',
            'сборник': 'problem_book'
        }

        for ru_type, en_type in type_mapping.items():
            if ru_type in source_type.lower():
                return en_type

        return source_type.lower()

    def _get_fallback_sources(self, topic: str) -> Dict[str, Any]:
        """Резервные источники при ошибке"""
        return {
            "topic": topic,
            "category": "general",
            "sources": {
                "online_course": [
                    {
                        "name": f"Курсы по {topic} на Coursera",
                        "type_with_emoji": "🎓 Онлайн-курс",
                        "level": "разный",
                        "description": f"Подборка курсов по теме '{topic}'",
                        "language": "смешанный"
                    }
                ],
                "textbook": [
                    {
                        "name": f"Учебники по {topic}",
                        "type_with_emoji": "📚 Учебник/Книга",
                        "level": "разный",
                        "description": f"Классические учебники по теме",
                        # "link": "",
                        "language": "смешанный"
                    }
                ]
            },
            "study_path": [
                "Начните с базовых понятий",
                "Изучите основные принципы",
                "Практикуйтесь на примерах"
            ],
            "recommendations": {
                "general": [
                    "Изучайте материал систематически",
                    "Делайте конспекты",
                    "Практикуйтесь регулярно"
                ]
            },
            "total_sources": 2
        }

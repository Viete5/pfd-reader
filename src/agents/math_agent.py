import os
import re
import logging
import time
from typing import Dict, Any, Optional
from langchain_gigachat.chat_models import GigaChat

from src.tools.pdf_math_indexer import extract_math_context_ultimate

try:
    from src.services.get_token import get_token as get_gigachat_credentials
except ImportError:
    try:
        # 2. Если не вышло (старая версия файла), пробуем старое имя
        from src.services.get_token import get_gigachat_credentials
    except ImportError:
        # 3. если совсем беда с импортами, берем из ENV напрямую
        import os
        def get_gigachat_credentials():
            return os.getenv("GIGACHAT_TOKEN", "")

logger = logging.getLogger(__name__)


class MathAgent:
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(), "solutions")
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_llm(self, temp=0.1):
        token = get_gigachat_credentials()
        return GigaChat(
            access_token=token,
            model="GigaChat",
            verify_ssl_certs=False,
            temperature=temp,
            scope="GIGACHAT_API_PERS"
        )

    def solve_task(self, task_spec: str, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"🚀 MathAgent: Решение задачи '{task_spec}'")
        try:
            if not os.path.exists(pdf_path):
                return {"success": False, "message": "PDF файл не найден."}

            # 1. Получаем Markdown
            md_text = extract_math_context_ultimate(pdf_path)

            if "CRITICAL_MARKER_ERROR" in md_text:
                return {"success": False, "message": f"Ошибка парсинга PDF: {md_text}"}

            # 2. Локализация контекста (грубый поиск)
            raw_context = self._locate_task_in_markdown(task_spec, md_text)

            if not raw_context:
                # Fallback
                match = re.search(r'(\d+)', task_spec)
                if match:
                    fallback_num = match.group(1)
                    idx = md_text.find(f"{fallback_num}.")
                    if idx != -1:
                        raw_context = md_text[idx:idx + 800]

            if not raw_context:
                return {"success": False, "message": f"Не удалось найти задачу '{task_spec}'."}

            logger.info(f"🎯 Контекст найден. Очищаю...")

            # 3. Выделение чистого условия (LLM)
            clean_condition = self._extract_clean_condition(task_spec, raw_context)

            # 4. Решение (структурированное)
            solution_latex = self._generate_structured_solution(task_spec, clean_condition)

            # 5. PDF
            pdf_file = self._render_pdf(task_spec, clean_condition, solution_latex)

            if pdf_file:
                return {"success": True, "pdf_path": pdf_file, "message": "Готово"}
            return {"success": False, "message": "Ошибка генерации PDF"}

        except Exception as e:
            logger.error(f"Agent Error: {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    def _locate_task_in_markdown(self, task_spec: str, md_text: str) -> Optional[str]:
        # 1. Извлекаем номер (например "8")
        match = re.search(r'(\d+)', task_spec)
        if not match: return None
        target_num = match.group(1)

        # 2. Ищем все вхождения этого числа, похожие на начало задачи
        # Ищем "Задача 8", "8.", "8)", "8 " в начале строки
        candidates = []
        lines = md_text.split('\n')

        for i, line in enumerate(lines):
            # Проверяем, начинается ли строка с нашего номера
            # (допускаем мусор в начале типа ">> 8.")
            if re.search(rf'(?:^|\s)(?:Задача\s*)?{target_num}\s*[\.\)]', line, re.IGNORECASE):
                # Берем контекст: саму строку и 20 строк после нее
                context_chunk = "\n".join(lines[i: i + 25])
                candidates.append(context_chunk)

        if not candidates:
            return None

        # 3. Возвращаем самый длинный кусок (на случай коллизий)
        return "\n---\n".join(candidates)

    def _extract_clean_condition(self, task_spec: str, raw_context: str) -> str:
        llm = self._get_llm(temp=0.1)

        prompt = f"""
        Ты — корректор математических текстов, восстановленных после плохого OCR.

        Твоя цель: Найти и восстановить условие задачи "{task_spec}" из фрагмента текста.

        ФРАГМЕНТ ТЕКСТА (содержит мусор и ошибки OCR):
        \"\"\"
        {raw_context[:2500]}
        \"\"\"

        ИНСТРУКЦИЯ:
        1. Найди текст, относящийся именно к задаче {task_spec}. Игнорируй соседние задачи.
        2. Восстанови формулы в LaTeX. Ошибки OCR исправляй по смыслу:
           - "xn" или "x n" -> $x_n$
           - "cn" -> $c^n$ (если это степень)
           - "lim n->00" -> $\\lim_{{n \\to \\infty}}$
           - "Vn" или "sqrt(n)" -> $\\sqrt{{n}}$
           - Дроби вида "a / b" -> $\\frac{{a}}{{b}}$
        3. Если в задаче несколько пунктов (а, б, в...), выбери тот, который указан в запросе (например, для "8а" бери только пункт а). Если буква не указана — выпиши ВСЕ пункты.

        Верни ТОЛЬКО восстановленный текст условия задачи. Никаких вступлений.
        """

        return llm.invoke(prompt).content.strip()

    def _generate_structured_solution(self, task_spec: str, condition: str) -> str:
        llm = self._get_llm(temp=0.2)
        prompt = f"""
        РОЛЬ: Профессор математики. Ты пишешь эталонное решение для студентов.
        ЗАДАЧА: "{task_spec}"
        УСЛОВИЕ:
        {condition}

        ТРЕБОВАНИЯ К РЕШЕНИЮ (Строго следуй структуре):
        1. **Теоретическая справка**: Кратко (1-2 предл.) объясни метод (например: "Для устранения неопределенности вида $\\infty/\\infty$ разделим числитель и знаменатель на старшую степень $n$").
        2. **Пошаговое решение**: Подробно распиши преобразования. Каждое действие — новая строка формул.
        3. **Ответ**: В конце.

        ФОРМАТ ВЫВОДА (LaTeX без преамбулы):
        \\subsection*{{Теория}}
        ...текст...
        \\subsection*{{Решение}}
        ...выкладки...
        \\begin{{equation*}}
        ...
        \\end{{equation*}}
        \\subsection*{{Ответ}}
        \\boxed{{...}}
        """
        return llm.invoke(prompt).content

    def _render_pdf(self, task_spec: str, condition: str, solution: str) -> Optional[str]:
        latex = r"""
        \documentclass[12pt]{article}
        \usepackage[utf8]{inputenc}
        \usepackage[T2A]{fontenc}
        \usepackage[russian]{babel}
        \usepackage{amsmath,amssymb}
        \usepackage{geometry}
        \geometry{a4paper, margin=2cm}
        \usepackage{parskip} % Отступы между абзацами

        \title{Решение задачи """ + task_spec + r"""}
        \author{MathAgent}
        \date{\today}

        \begin{document}
        \maketitle

        \section*{Условие}
        """ + condition + r"""

        \hrulefill

        """ + solution + r"""

        \end{document}
        """

        import subprocess, tempfile, shutil
        try:
            wd = tempfile.mkdtemp()
            tex = os.path.join(wd, "sol.tex")
            with open(tex, "w", encoding="utf-8") as f:
                f.write(latex)

            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", wd, tex],
                           stdout=subprocess.DEVNULL, timeout=20)

            if os.path.exists(os.path.join(wd, "sol.pdf")):
                dst = os.path.join(self.output_dir, f"Sol_{int(time.time())}.pdf")
                shutil.copy(os.path.join(wd, "sol.pdf"), dst)
                shutil.rmtree(wd)
                return dst
        except Exception as e:
            logger.error(f"LaTeX Error: {e}")
            return None

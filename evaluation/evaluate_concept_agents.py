import asyncio
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
import pandas as pd


class ConceptAgentEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer("cointegrated/rubert-tiny2")

    async def evaluate_concept_extraction(self, test_texts: List[Dict]) -> Dict:
        """
        Оценка извлечения концептов
        """
        from src.agents.concept_explainer import ConceptExplainerAgent

        agent = ConceptExplainerAgent()
        results = []

        for test_case in test_texts:
            text = test_case["text"]
            expected_concepts = test_case["expected_concepts"]

            try:
                extracted_concepts = agent.extract_concepts(text, max_concepts=10)

                # Преобразуем в множества для сравнения
                extracted_names = {c["name"].lower() for c in extracted_concepts}
                expected_names = {c.lower() for c in expected_concepts}

                # Вычисляем метрики
                precision = len(extracted_names & expected_names) / max(len(extracted_names), 1)
                recall = len(extracted_names & expected_names) / max(len(expected_names), 1)
                f1 = 2 * precision * recall / max((precision + recall), 0.001)

                # Семантическое сходство
                semantic_similarity = self._calculate_concept_similarity(
                    list(extracted_names), list(expected_names)
                )

                results.append({
                    "text_id": test_case.get("id", len(results)),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "semantic_similarity": semantic_similarity,
                    "num_extracted": len(extracted_names),
                    "num_expected": len(expected_names)
                })

            except Exception as e:
                results.append({
                    "text_id": test_case.get("id", len(results)),
                    "error": str(e),
                    "precision": 0,
                    "recall": 0,
                    "f1": 0
                })

        # Агрегируем результаты
        if results:
            df = pd.DataFrame(results)
            return {
                "avg_precision": df["precision"].mean(),
                "avg_recall": df["recall"].mean(),
                "avg_f1": df["f1"].mean(),
                "avg_semantic_similarity": df["semantic_similarity"].mean(),
                "extraction_rate": len([r for r in results if r["num_extracted"] > 0]) / len(results),
                "total_tests": len(results)
            }

        return {"error": "No results"}

    def _calculate_concept_similarity(self, extracted: List[str], expected: List[str]) -> float:
        """Вычисляет семантическое сходство между наборами концептов"""
        if not extracted or not expected:
            return 0.0

        # Эмбеддинги для всех концептов
        all_concepts = extracted + expected
        embeddings = self.embedding_model.encode(all_concepts)

        # Матрица сходства
        extracted_embeddings = embeddings[:len(extracted)]
        expected_embeddings = embeddings[len(extracted):]

        similarity_matrix = np.dot(extracted_embeddings, expected_embeddings.T)

        # Максимальное сходство для каждого извлеченного концепта
        max_similarities = np.max(similarity_matrix, axis=1)

        return float(np.mean(max_similarities))

    async def evaluate_concept_explanation(self, test_concepts: List[Dict]) -> Dict:
        """
        Оценка качества объяснения концептов
        """
        from src.agents.concept_explainer import ConceptExplainerAgent

        agent = ConceptExplainerAgent()
        results = []

        for test_case in test_concepts:
            concept = test_case["concept"]
            expected_explanation = test_case.get("expected_explanation", "")

            try:
                explanation_result = agent.explain_concept(concept)

                if explanation_result and "explanation" in explanation_result:
                    explanation = explanation_result["explanation"]

                    # Оценка качества объяснения
                    clarity_score = self._evaluate_clarity(explanation)
                    completeness_score = self._evaluate_completeness(explanation, concept)
                    structure_score = self._evaluate_structure(explanation)

                    # Семантическое сходство с ожидаемым
                    semantic_similarity = 0.5
                    if expected_explanation:
                        semantic_similarity = self._calculate_text_similarity(
                            explanation, expected_explanation
                        )

                    results.append({
                        "concept": concept,
                        "clarity": clarity_score,
                        "completeness": completeness_score,
                        "structure": structure_score,
                        "semantic_similarity": semantic_similarity,
                        "length": len(explanation),
                        "has_examples": "examples" in explanation_result,
                        "has_key_points": "key_points" in explanation_result
                    })

            except Exception as e:
                results.append({
                    "concept": concept,
                    "error": str(e),
                    "clarity": 0,
                    "completeness": 0,
                    "structure": 0
                })

        # Агрегируем результаты
        if results:
            df = pd.DataFrame(results)
            return {
                "avg_clarity": df["clarity"].mean(),
                "avg_completeness": df["completeness"].mean(),
                "avg_structure": df["structure"].mean(),
                "avg_semantic_similarity": df["semantic_similarity"].mean(),
                "explanation_rate": len([r for r in results if "error" not in r]) / len(results),
                "examples_rate": df["has_examples"].mean() if "has_examples" in df.columns else 0,
                "key_points_rate": df["has_key_points"].mean() if "has_key_points" in df.columns else 0
            }

        return {"error": "No results"}

    def _evaluate_clarity(self, text: str) -> float:
        """Оценка ясности и понятности текста"""
        # Проверяем наличие сложных конструкций
        complex_patterns = [
            r'\bоднако\b', r'\bследовательно\b', r'\bтаким образом\b',
            r'\bввиду того что\b', r'\bнесмотря на то что\b'
        ]

        # Длина предложений (короткие предложения обычно понятнее)
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Оценка: меньше сложных конструкций и умеренная длина предложений - лучше
        complexity_penalty = sum(1 for pattern in complex_patterns
                                 if len(re.findall(pattern, text.lower())) > 0) / len(complex_patterns)

        sentence_length_score = 1.0 - min(1.0, abs(avg_sentence_length - 15) / 15)

        return 0.6 * sentence_length_score + 0.4 * (1 - complexity_penalty)

    def _evaluate_completeness(self, explanation: str, concept: str) -> float:
        """Оценка полноты объяснения"""
        # Ключевые аспекты, которые должны быть в хорошем объяснении
        required_aspects = [
            "определение", "пример", "применение",
            "принцип", "характеристика", "значение"
        ]

        # Проверяем наличие этих аспектов в тексте
        explanation_lower = explanation.lower()
        aspect_coverage = sum(1 for aspect in required_aspects
                              if any(word in explanation_lower for word in aspect.split()))

        return aspect_coverage / len(required_aspects)

    def _evaluate_structure(self, text: str) -> float:
        """Оценка структурированности текста"""
        # Маркеры структуры
        structure_markers = [
            'во-первых', 'во-вторых', 'во-третьих',
            'с одной стороны', 'с другой стороны',
            'например', 'таким образом', 'в заключение',
            'важно отметить', 'следует подчеркнуть'
        ]

        # Количество абзацев (по переносам строк)
        paragraphs = text.split('\n\n')

        # Подсчет маркеров структуры
        markers_found = sum(1 for marker in structure_markers
                            if marker in text.lower())

        # Оценка структуры
        structure_score = min(1.0, markers_found / 3) * 0.4  # За маркеры
        paragraph_score = min(1.0, len(paragraphs) / 3) * 0.6  # За абзацы

        return structure_score + paragraph_score

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет семантическое сходство между текстами"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)


async def evaluate_study_advisor():
    """Оценка агента учебных советов"""
    from src.agents.study_advisor import StudyAdvisorAgent

    agent = StudyAdvisorAgent()
    results = {}

    try:
        # Тестирование разных функций агента
        general_advice = agent.get_study_advice()
        notes_advice = agent.get_notes_advice("физика")
        memory_techniques = agent.get_memory_techniques()

        # Оценка структуры ответов
        results["general_advice_has_structure"] = all(
            key in general_advice for key in ["advice", "quick_tips", "methods"]
        )
        results["notes_advice_has_structure"] = all(
            key in notes_advice for key in ["advice", "techniques", "tools"]
        )
        results["memory_techniques_has_structure"] = all(
            key in memory_techniques for key in ["advice", "techniques", "exercises"]
        )

        # Оценка полезности
        evaluator = ConceptAgentEvaluator()
        results["general_advice_clarity"] = evaluator._evaluate_clarity(
            general_advice.get("advice", "")
        )
        results["notes_advice_clarity"] = evaluator._evaluate_clarity(
            notes_advice.get("advice", "")
        )

        # Оценка полноты
        results["general_advice_completeness"] = len(general_advice.get("quick_tips", [])) > 2
        results["notes_advice_completeness"] = len(notes_advice.get("techniques", [])) > 2

    except Exception as e:
        results["error"] = str(e)

    return results


async def run_concept_agent_evaluation():
    """Запуск оценки агентов концептов"""

    print("🧠 Начинаю оценку агентов концептов...")

    # Тестовые данные
    test_texts = [
        {
            "id": 1,
            "text": "Физика изучает законы природы. Основные разделы: механика, термодинамика, электродинамика. Ньютон открыл законы движения.",
            "expected_concepts": ["физика", "механика", "термодинамика", "электродинамика", "ньютон"]
        },
        {
            "id": 2,
            "text": "Математика включает алгебру, геометрию, анализ. Дифференциальное исчисление изучает производные.",
            "expected_concepts": ["математика", "алгебра", "геометрия", "анализ", "дифференциальное исчисление"]
        }
    ]

    test_concepts = [
        {
            "concept": "гравитация",
            "expected_explanation": "Гравитация - это сила притяжения между объектами с массой"
        },
        {
            "concept": "интеграл",
            "expected_explanation": "Интеграл - это математическая операция, обратная дифференцированию"
        }
    ]

    evaluator = ConceptAgentEvaluator()

    # Оценка извлечения концептов
    print("\n1. Оценка извлечения концептов:")
    extraction_metrics = await evaluator.evaluate_concept_extraction(test_texts)
    for key, value in extraction_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Оценка объяснения концептов
    print("\n2. Оценка объяснения концептов:")
    explanation_metrics = await evaluator.evaluate_concept_explanation(test_concepts)
    for key, value in explanation_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Оценка Study Advisor
    print("\n3. Оценка Study Advisor:")
    study_advisor_metrics = await evaluate_study_advisor()
    for key, value in study_advisor_metrics.items():
        print(f"   {key}: {value}")

    return {
        "concept_extraction": extraction_metrics,
        "concept_explanation": explanation_metrics,
        "study_advisor": study_advisor_metrics
    }


if __name__ == "__main__":
    asyncio.run(run_concept_agent_evaluation())

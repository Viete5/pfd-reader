import asyncio
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from src.tools.rag_query import RAGLoader


class RAGEvaluator:
    def __init__(self, embedding_model: str = "cointegrated/rubert-tiny2"):
        self.embedding_model = SentenceTransformer(embedding_model)

    async def evaluate_retrieval_quality(self, user_id: int, test_queries: List[Dict]) -> Dict:
        """
        Оценка качества ретривера
        """
        try:
            loader = RAGLoader(user_id)
            retriever = loader.retriever

            metrics = {
                "precision_at_k": [],
                "recall_at_k": [],
                "mrr": [],
                "hit_rate": []
            }

            for query_data in test_queries:
                query = query_data["query"]
                relevant_docs = query_data["relevant_docs"]

                # Получаем результаты ретривера
                retrieved_docs = retriever.get_relevant_documents(query)
                retrieved_texts = [doc.page_content for doc in retrieved_docs]

                # Вычисляем метрики
                precision_k = self._calculate_precision_at_k(retrieved_texts, relevant_docs, k=3)
                recall_k = self._calculate_recall_at_k(retrieved_texts, relevant_docs, k=3)
                mrr = self._calculate_mrr(retrieved_texts, relevant_docs)
                hit_rate = self._calculate_hit_rate(retrieved_texts, relevant_docs)

                metrics["precision_at_k"].append(precision_k)
                metrics["recall_at_k"].append(recall_k)
                metrics["mrr"].append(mrr)
                metrics["hit_rate"].append(hit_rate)

            # Агрегируем результаты
            return {
                "precision@3_mean": np.mean(metrics["precision_at_k"]),
                "precision@3_std": np.std(metrics["precision_at_k"]),
                "recall@3_mean": np.mean(metrics["recall_at_k"]),
                "recall@3_std": np.std(metrics["recall_at_k"]),
                "mrr_mean": np.mean(metrics["mrr"]),
                "hit_rate_mean": np.mean(metrics["hit_rate"]),
                "num_queries": len(test_queries)
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_precision_at_k(self, retrieved: List[str], relevant: List[str], k: int = 3) -> float:
        """Precision@K - точность первых K результатов"""
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            return 0.0

        # Используем семантическое сходство вместо точного совпадения
        relevant_count = 0
        for ret_doc in retrieved_k:
            if any(self._semantic_similarity(ret_doc, rel_doc) > 0.6 for rel_doc in relevant):
                relevant_count += 1

        return relevant_count / k

    def _calculate_recall_at_k(self, retrieved: List[str], relevant: List[str], k: int = 3) -> float:
        """Recall@K - полнота первых K результатов"""
        retrieved_k = retrieved[:k]
        if not relevant:
            return 0.0

        # Находим, сколько релевантных документов найдено
        found_relevant = set()
        for ret_doc in retrieved_k:
            for i, rel_doc in enumerate(relevant):
                if self._semantic_similarity(ret_doc, rel_doc) > 0.6:
                    found_relevant.add(i)

        return len(found_relevant) / len(relevant)

    def _calculate_mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank - среднее обратное ранжирование"""
        for rank, ret_doc in enumerate(retrieved, 1):
            for rel_doc in relevant:
                if self._semantic_similarity(ret_doc, rel_doc) > 0.6:
                    return 1.0 / rank
        return 0.0

    def _calculate_hit_rate(self, retrieved: List[str], relevant: List[str]) -> float:
        """Hit Rate - процент запросов, где найден хотя бы один релевантный документ"""
        for ret_doc in retrieved:
            for rel_doc in relevant:
                if self._semantic_similarity(ret_doc, rel_doc) > 0.6:
                    return 1.0
        return 0.0

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет семантическое сходство между текстами"""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    async def evaluate_response_quality(self, user_id: int, test_cases: List[Dict]) -> Dict:
        """
        Оценка качества ответов LLM
        """
        from src.agents.RAG import RAGAgent

        rag_agent = RAGAgent()
        results = []

        for test_case in test_cases:
            query = test_case["query"]
            expected_answer = test_case.get("expected_answer", "")

            try:
                response = await asyncio.to_thread(rag_agent.run, user_id, query)

                # Оценка с использованием метрик
                evaluation = {
                    "query": query,
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "length": len(response),
                    "relevance_score": self._calculate_relevance(query, response, expected_answer),
                    "factuality_score": self._calculate_factuality(response, expected_answer),
                    "coherence_score": self._calculate_coherence(response),
                    "contains_keywords": self._check_keywords(response, query)
                }
                results.append(evaluation)

            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "relevance_score": 0,
                    "factuality_score": 0,
                    "coherence_score": 0
                })

        # Агрегируем результаты
        if results:
            df = pd.DataFrame(results)
            return {
                "avg_relevance": df["relevance_score"].mean(),
                "avg_factuality": df["factuality_score"].mean(),
                "avg_coherence": df["coherence_score"].mean(),
                "avg_response_length": df["length"].mean(),
                "total_queries": len(results),
                "success_rate": len([r for r in results if "error" not in r]) / len(results)
            }
        return {"error": "No results"}

    def _calculate_relevance(self, query: str, response: str, expected: str) -> float:
        """Оценка релевантности ответа запросу"""
        query_embedding = self.embedding_model.encode(query)
        response_embedding = self.embedding_model.encode(response)

        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]

        # Дополнительная проверка по ключевым словам
        query_keywords = set(query.lower().split())
        response_keywords = set(response.lower().split())
        keyword_overlap = len(query_keywords.intersection(response_keywords)) / max(len(query_keywords), 1)

        # Комбинированная оценка
        return 0.6 * similarity + 0.3 * keyword_overlap

    def _calculate_factuality(self, response: str, expected: str) -> float:
        """Оценка фактической точности (упрощенная)"""
        if not expected:
            return 0.5  # Невозможно проверить

        response_embedding = self.embedding_model.encode(response)
        expected_embedding = self.embedding_model.encode(expected)

        similarity = cosine_similarity([response_embedding], [expected_embedding])[0][0]
        return similarity

    def _calculate_coherence(self, response: str) -> float:
        """Оценка связности и структурированности ответа"""
        # Проверяем структуру ответа
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.3

        # Проверяем наличие маркеров структуры
        structure_indicators = ['во-первых', 'во-вторых', 'с одной стороны',
                                'с другой стороны', 'например', 'таким образом']

        has_structure = any(indicator in response.lower() for indicator in structure_indicators)

        # Проверяем длину предложений (слишком длинные или короткие - плохо)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        sentence_length_score = 1.0 - abs(avg_sentence_length - 15) / 15  # Идеал 15 слов

        return 0.3 * has_structure + 0.6 * min(1.0, sentence_length_score)

    def _check_keywords(self, response: str, query: str) -> List[str]:
        """Проверяет наличие ключевых слов из запроса в ответе"""
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        response_words = set(word.lower() for word in response.split() if len(word) > 3)

        return list(query_words.intersection(response_words))


async def run_evaluation(user_id: int = 12345):
    """Запуск полной оценки системы"""

    # Тестовые данные для оценки
    test_queries = [
        {
            "query": "Что такое второй закон Ньютона?",
            "relevant_docs": [
                "Второй закон Ньютона: F = ma, сила равна массе умноженной на ускорение",
                "Ньютон установил, что сила пропорциональна ускорению тела"
            ]
        },
        {
            "query": "Найди информацию о квантовой физике",
            "relevant_docs": [
                "Квантовая физика изучает поведение микрочастиц",
                "Основные принципы: суперпозиция, запутанность, неопределенность"
            ]
        }
    ]

    test_cases = [
        {
            "query": "Объясни что такое гравитация",
            "expected_answer": "Гравитация - это сила притяжения между объектами с массой"
        },
        {
            "query": "Какие бывают методы решения уравнений?",
            "expected_answer": "Аналитические, численные, графические методы решения уравнений"
        }
    ]

    evaluator = RAGEvaluator()

    print("🔍 Начинаю оценку RAG системы...")

    # Оценка ретривера
    print("\n1. Оценка качества ретривера:")
    retrieval_metrics = await evaluator.evaluate_retrieval_quality(user_id, test_queries)
    for key, value in retrieval_metrics.items():
        print(f"   {key}: {value:.3f}")

    # Оценка ответов
    print("\n2. Оценка качества ответов LLM:")
    response_metrics = await evaluator.evaluate_response_quality(user_id, test_cases)
    for key, value in response_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    return {
        "retrieval_metrics": retrieval_metrics,
        "response_metrics": response_metrics
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation())


import asyncio
import time
import psutil
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class SystemMetricsEvaluator:
    def __init__(self):
        self.metrics_history = []

    async def measure_performance(self, user_id: int, queries: List[str]) -> Dict:
        """
        Измерение производительности системы
        """
        from src.core.orchestrator import handle_user_query

        results = []

        for query in queries:
            try:
                # Измеряем время выполнения
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                response = await handle_user_query(user_id, query)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory

                results.append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "execution_time": execution_time,
                    "memory_usage_mb": end_memory,
                    "memory_delta_mb": memory_delta,
                    "response_length": len(response),
                    "success": True
                })

            except Exception as e:
                results.append({
                    "query": query[:50] + "...",
                    "error": str(e),
                    "execution_time": 0,
                    "success": False
                })

        # Агрегируем метрики
        if results:
            df = pd.DataFrame([r for r in results if r["success"]])

            if not df.empty:
                return {
                    "avg_execution_time": df["execution_time"].mean(),
                    "max_execution_time": df["execution_time"].max(),
                    "min_execution_time": df["execution_time"].min(),
                    "throughput_qps": 1 / df["execution_time"].mean() if df["execution_time"].mean() > 0 else 0,
                    "avg_memory_usage_mb": df["memory_usage_mb"].mean(),
                    "avg_memory_delta_mb": df["memory_delta_mb"].mean(),
                    "success_rate": len(df) / len(results),
                    "total_queries": len(results)
                }

        return {"error": "No successful queries"}

    async def load_test(self, user_id: int, num_concurrent: int = 10) -> Dict:
        """
        Нагрузочное тестирование системы
        """
        from src.core.orchestrator import handle_user_query

        test_queries = [
                           "Что такое физика?",
                           "Объясни закон Ома",
                           "Найди материалы по математике",
                           "Дай учебные советы",
                           "Как улучшить конспекты?"
                       ] * (num_concurrent // 2)  # Увеличиваем количество запросов

        async def run_query(query):
            try:
                start_time = time.time()
                await handle_user_query(user_id, query)
                return time.time() - start_time
            except Exception as e:
                return None

        # Запускаем запросы параллельно
        start_time = time.time()
        tasks = [run_query(query) for query in test_queries[:num_concurrent]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Анализируем результаты
        successful_times = [r for r in results if isinstance(r, (int, float))]

        return {
            "total_time_seconds": total_time,
            "queries_per_second": len(successful_times) / total_time if total_time > 0 else 0,
            "avg_response_time": np.mean(successful_times) if successful_times else 0,
            "p95_response_time": np.percentile(successful_times, 95) if successful_times else 0,
            "success_rate": len(successful_times) / len(results) if results else 0,
            "concurrent_users": num_concurrent,
            "total_queries": len(results)
        }

    async def evaluate_scalability(self, user_ids: List[int]) -> Dict:
        """
        Оценка масштабируемости системы
        """
        test_query = "Что такое гравитация?"
        results = []

        for num_users in [1, 3, 5, 10]:
            start_time = time.time()

            # Создаем задачи для каждого пользователя
            tasks = []
            for i in range(min(num_users, len(user_ids))):
                user_id = user_ids[i]
                tasks.append(self._single_user_query(user_id, test_query))

            # Запускаем параллельно
            await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            results.append({
                "num_users": num_users,
                "total_time": total_time,
                "throughput": num_users / total_time if total_time > 0 else 0
            })

        # Анализируем масштабируемость
        df = pd.DataFrame(results)
        scalability_factor = df["throughput"].iloc[-1] / df["throughput"].iloc[0] if len(df) > 1 else 1

        return {
            "scalability_results": results,
            "scalability_factor": scalability_factor,
            "is_linear_scaling": scalability_factor > 0.7 * (df["num_users"].iloc[-1] / df["num_users"].iloc[0])
        }

    async def _single_user_query(self, user_id: int, query: str):
        """Вспомогательная функция для запроса от одного пользователя"""
        from src.core.orchestrator import handle_user_query
        try:
            await handle_user_query(user_id, query)
        except:
            pass

    def collect_resource_metrics(self) -> Dict:
        """Сбор метрик использования ресурсов"""
        process = psutil.Process()

        return {
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_percent": process.memory_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "disk_io": process.io_counters()
        }

    async def run_comprehensive_evaluation(self, user_id: int = 12345):
        """Комплексная оценка системы"""

        print("⚡ Начинаю системную оценку...")

        # Тестовые запросы
        test_queries = [
            "Что такое физика?",
            "Объясни второй закон Ньютона",
            "Найди учебники по математике",
            "Как эффективно учиться?",
            "Улучши мой конспект",
            "Создай учебный план на неделю"
        ]

        # 1. Производительность
        print("\n1. Оценка производительности:")
        perf_metrics = await self.measure_performance(user_id, test_queries)
        for key, value in perf_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        # 2. Нагрузочное тестирование
        print("\n2. Нагрузочное тестирование (10 пользователей):")
        load_metrics = await self.load_test(user_id, num_concurrent=10)
        for key, value in load_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        # 3. Масштабируемость
        print("\n3. Оценка масштабируемости:")
        user_ids = [12345, 12346, 12347, 12348, 12349]
        scalability_metrics = await self.evaluate_scalability(user_ids)
        print(f"   Фактор масштабируемости: {scalability_metrics['scalability_factor']:.3f}")
        print(f"   Линейное масштабирование: {scalability_metrics['is_linear_scaling']}")

        # 4. Использование ресурсов
        print("\n4. Использование ресурсов:")
        resource_metrics = self.collect_resource_metrics()
        for key, value in resource_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            elif hasattr(value, '_asdict'):
                print(f"   {key}: {value._asdict()}")
            else:
                print(f"   {key}: {value}")

        # Сводный отчет
        print("\n📊 СВОДНЫЙ ОТЧЕТ:")
        summary = {
            "performance_score": self._calculate_performance_score(perf_metrics),
            "scalability_score": scalability_metrics["scalability_factor"] * 10,
            "reliability_score": load_metrics.get("success_rate", 0) * 100,
            "resource_efficiency": 100 - resource_metrics.get("memory_percent", 0)
        }

        for key, value in summary.items():
            print(f"   {key}: {value:.1f}/100")

        overall_score = np.mean(list(summary.values()))
        print(f"\n🎯 ОБЩАЯ ОЦЕНКА СИСТЕМЫ: {overall_score:.1f}/100")

        # Рекомендации по улучшению
        self._generate_recommendations(summary)

        return {
            "performance": perf_metrics,
            "load_test": load_metrics,
            "scalability": scalability_metrics,
            "resources": resource_metrics,
            "summary": summary,
            "overall_score": overall_score
        }

    def _calculate_performance_score(self, perf_metrics: Dict) -> float:
        """Рассчитывает оценку производительности"""
        score = 100

        # Штраф за медленные запросы
        if perf_metrics.get("avg_execution_time", 10) > 5:
            score -= 30
        elif perf_metrics.get("avg_execution_time", 10) > 2:
            score -= 15

        # Бонус за высокую пропускную способность
        if perf_metrics.get("throughput_qps", 0) > 1:
            score += 10

        # Штраф за высокое использование памяти
        if perf_metrics.get("avg_memory_usage_mb", 1000) > 500:
            score -= 20

        return max(0, min(100, score))

    def _generate_recommendations(self, summary: Dict):
        """Генерирует рекомендации по улучшению"""
        print("\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")

        recommendations = []

        if summary["performance_score"] < 70:
            recommendations.append("• Оптимизировать время ответа RAG системы")
            recommendations.append("• Кэшировать частые запросы")

        if summary["scalability_score"] < 70:
            recommendations.append("• Улучшить параллельную обработку запросов")
            recommendations.append("• Рассмотреть использование пулов соединений")

        if summary["resource_efficiency"] < 70:
            recommendations.append("• Оптимизировать использование памяти")
            recommendations.append("• Реализовать очистку неиспользуемых сессий")

        if not recommendations:
            recommendations.append("• Система работает хорошо! Продолжайте мониторить метрики")

        for rec in recommendations:
            print(f"   {rec}")


async def main():
    evaluator = SystemMetricsEvaluator()
    results = await evaluator.run_comprehensive_evaluation()
    return results


if __name__ == "__main__":
    asyncio.run(main())

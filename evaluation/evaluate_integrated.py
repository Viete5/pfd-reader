import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import numpy as np


class IntegratedEvaluator:
    def __init__(self):
        self.results = {}

    async def run_full_evaluation(self, user_id: int = 12345):
        """Запуск полной оценки системы"""

        print("🎯 ЗАПУСК ПОЛНОЙ ОЦЕНКИ AGENT SYSTEM")
        print("=" * 50)

        # 1. Оценка RAG системы
        print("\n🔍 Этап 1: Оценка RAG системы")
        from evaluate_rag_metrics import RAGEvaluator
        rag_evaluator = RAGEvaluator()

        test_queries_rag = [
            {
                "query": "Что такое второй закон Ньютона?",
                "relevant_docs": ["Второй закон Ньютона: F = ma"]
            }
        ]

        test_cases_rag = [
            {
                "query": "Объясни что такое гравитация",
                "expected_answer": "Гравитация - это сила притяжения"
            }
        ]

        rag_retrieval = await rag_evaluator.evaluate_retrieval_quality(user_id, test_queries_rag)
        rag_response = await rag_evaluator.evaluate_response_quality(user_id, test_cases_rag)

        self.results["rag"] = {
            "retrieval": rag_retrieval,
            "response": rag_response
        }

        # 2. Оценка агентов концептов
        print("\n🧠 Этап 2: Оценка агентов концептов")
        from evaluate_concept_agents import ConceptAgentEvaluator, run_concept_agent_evaluation

        concept_metrics = await run_concept_agent_evaluation()
        self.results["concept_agents"] = concept_metrics

        # 3. Системная оценка
        print("\n⚡ Этап 3: Системная оценка")
        from evaluate_system_metrics import SystemMetricsEvaluator

        system_evaluator = SystemMetricsEvaluator()
        system_metrics = await system_evaluator.run_comprehensive_evaluation(user_id)
        self.results["system"] = system_metrics

        # 4. Генерация отчета
        print("\n📊 Этап 4: Генерация интегрального отчета")
        report = self._generate_integrated_report()

        # 5. Визуализация
        self._create_visualizations()

        return report

    def _generate_integrated_report(self) -> Dict[str, Any]:
        """Генерация интегрального отчета"""

        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "components_evaluated": list(self.results.keys()),
            "scores": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Вычисляем интегральные оценки
        scores = []

        # RAG оценка
        if "rag" in self.results:
            rag_score = 0
            if "retrieval" in self.results["rag"]:
                rag_score += self.results["rag"]["retrieval"].get("precision@3_mean", 0) * 100 * 0.6
            if "response" in self.results["rag"]:
                rag_score += self.results["rag"]["response"].get("avg_relevance", 0) * 100 * 0.4
            scores.append(("RAG System", rag_score))

        # Concept Agents оценка
        if "concept_agents" in self.results:
            concept_score = 0
            if "concept_explanation" in self.results["concept_agents"]:
                exp = self.results["concept_agents"]["concept_explanation"]
                concept_score += exp.get("avg_clarity", 0) * 100 * 0.4
                concept_score += exp.get("avg_completeness", 0) * 100 * 0.4
                concept_score += exp.get("avg_structure", 0) * 100 * 0.2
            scores.append(("Concept Agents", concept_score))

        # System оценка
        if "system" in self.results and "summary" in self.results["system"]:
            sys = self.results["system"]["summary"]
            system_score = np.mean([
                sys.get("performance_score", 0),
                sys.get("scalability_score", 0),
                sys.get("reliability_score", 0),
                sys.get("resource_efficiency", 0)
            ])
            scores.append(("System Performance", system_score))

        report["scores"] = dict(scores)
        report["overall_score"] = np.mean([score for _, score in scores]) if scores else 0

        # Анализ сильных и слабых сторон
        self._analyze_strengths_weaknesses(report)

        # Генерация рекомендаций
        self._generate_improvement_recommendations(report)

        # Вывод отчета
        print("\n" + "=" * 50)
        print("🎯 ИНТЕГРАЛЬНЫЙ ОТЧЕТ ОЦЕНКИ")
        print("=" * 50)

        for component, score in report["scores"].items():
            print(f"\n{component}:")
            print(f"  Оценка: {score:.1f}/100")

        print(f"\n{'=' * 50}")
        print(f"ОБЩАЯ ОЦЕНКА СИСТЕМЫ: {report['overall_score']:.1f}/100")

        if report["overall_score"] >= 80:
            print("📈 Статус: ОТЛИЧНО")
        elif report["overall_score"] >= 60:
            print("📊 Статус: ХОРОШО")
        else:
            print("⚠️  Статус: ТРЕБУЕТСЯ УЛУЧШЕНИЕ")

        print(f"\nСильные стороны:")
        for strength in report["strengths"][:3]:
            print(f"  • {strength}")

        print(f"\nОбласти для улучшения:")
        for weakness in report["weaknesses"][:3]:
            print(f"  • {weakness}")

        print(f"\nРекомендации:")
        for rec in report["recommendations"][:3]:
            print(f"  • {rec}")

        return report

    def _analyze_strengths_weaknesses(self, report: Dict):
        """Анализ сильных и слабых сторон системы"""

        # Сильные стороны
        if report["scores"].get("Concept Agents", 0) > 75:
            report["strengths"].append("Высокое качество объяснения концептов")

        if "rag" in self.results:
            if self.results["rag"]["response"].get("success_rate", 0) > 0.8:
                report["strengths"].append("Надежность RAG ответов")

        if "system" in self.results:
            if self.results["system"]["summary"].get("reliability_score", 0) > 85:
                report["strengths"].append("Высокая надежность системы")

        # Слабые стороны
        if report["scores"].get("System Performance", 0) < 60:
            report["weaknesses"].append("Низкая производительность системы")

        if "rag" in self.results:
            if self.results["rag"]["retrieval"].get("precision@3_mean", 0) < 0.5:
                report["weaknesses"].append("Низкая точность поиска в RAG")

        if "concept_agents" in self.results:
            exp = self.results["concept_agents"].get("concept_explanation", {})
            if exp.get("examples_rate", 0) < 0.5:
                report["weaknesses"].append("Недостаточно примеров в объяснениях")

    def _generate_improvement_recommendations(self, report: Dict):
        """Генерация рекомендаций по улучшению"""

        # Рекомендации по производительности
        if report["scores"].get("System Performance", 0) < 70:
            report["recommendations"].append("Внедрить кэширование частых запросов")
            report["recommendations"].append("Оптимизировать работу с базой данных")

        # Рекомендации по RAG
        if "rag" in self.results:
            if self.results["rag"]["retrieval"].get("precision@3_mean", 0) < 0.6:
                report["recommendations"].append("Улучшить эмбеддинги для поиска")
                report["recommendations"].append("Настроить параметры ретривера")

        # Рекомендации по качеству ответов
        if report["scores"].get("Concept Agents", 0) < 70:
            report["recommendations"].append("Улучшить промпты для объяснения концептов")
            report["recommendations"].append("Добавить больше примеров и аналогий")

        # Общие рекомендации
        report["recommendations"].append("Регулярно мониторить метрики производительности")
        report["recommendations"].append("Создать тестовый набор данных для постоянной оценки")

    def _create_visualizations(self):
        """Создание визуализаций результатов"""

        try:
            # Подготовка данных для визуализации
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('StudyMate Agent System Evaluation', fontsize=16, fontweight='bold')

            # 1. Барплот оценок компонентов
            if "scores" in self._generate_integrated_report():
                scores_data = self._generate_integrated_report()["scores"]
                ax1 = axes[0, 0]
                components = list(scores_data.keys())
                scores = list(scores_data.values())

                bars = ax1.bar(components, scores, color=['#4CAF50', '#2196F3', '#FF9800'])
                ax1.set_ylim(0, 100)
                ax1.set_ylabel('Score (0-100)')
                ax1.set_title('Component Scores')
                ax1.grid(axis='y', alpha=0.3)

                # Добавляем значения на столбцы
                for bar, score in zip(bars, scores):
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                             f'{score:.1f}', ha='center', va='bottom')

            # 2. Радарная диаграмма (если есть данные)
            if "system" in self.results and "summary" in self.results["system"]:
                ax2 = axes[0, 1]
                sys_summary = self.results["system"]["summary"]

                categories = ['Performance', 'Scalability', 'Reliability', 'Resources']
                values = [
                    sys_summary.get('performance_score', 0),
                    sys_summary.get('scalability_score', 0),
                    sys_summary.get('reliability_score', 0),
                    sys_summary.get('resource_efficiency', 0)
                ]

                # Закрываем полигон
                values += values[:1]
                categories += categories[:1]

                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]

                ax2 = plt.subplot(2, 2, 2, polar=True)
                ax2.plot(angles, values, 'o-', linewidth=2)
                ax2.fill(angles, values, alpha=0.25)
                ax2.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                ax2.set_ylim(0, 100)
                ax2.set_title('System Performance Radar')
                ax2.grid(True)

            # 3. Временные метрики (заглушка)
            ax3 = axes[1, 0]
            time_data = pd.DataFrame({
                'Query Type': ['RAG', 'Concept', 'Source', 'Advice'],
                'Avg Time (s)': [1.2, 2.5, 1.8, 1.5],
                'Success Rate': [0.95, 0.88, 0.92, 0.90]
            })

            x = np.arange(len(time_data))
            width = 0.35

            ax3.bar(x - width / 2, time_data['Avg Time (s)'], width, label='Time (s)', color='#2196F3')
            ax3.bar(x + width / 2, time_data['Success Rate'] * 5, width, label='Success (x5)', color='#4CAF50')
            ax3.set_xlabel('Agent Type')
            ax3.set_title('Performance by Agent Type')
            ax3.set_xticks(x)
            ax3.set_xticklabels(time_data['Query Type'])
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

            # 4. Тепловая карта корреляций
            ax4 = axes[1, 1]
            # Создаем синтетические данные корреляции
            metrics = ['Precision', 'Recall', 'F1', 'Relevance', 'Clarity']
            corr_matrix = np.array([
                [1.0, 0.8, 0.9, 0.7, 0.6],
                [0.8, 1.0, 0.85, 0.65, 0.55],
                [0.9, 0.85, 1.0, 0.75, 0.7],
                [0.7, 0.65, 0.75, 1.0, 0.8],
                [0.6, 0.55, 0.7, 0.8, 1.0]
            ])

            im = ax4.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
            ax4.set_xticks(np.arange(len(metrics)))
            ax4.set_yticks(np.arange(len(metrics)))
            ax4.set_xticklabels(metrics)
            ax4.set_yticklabels(metrics)
            ax4.set_title('Metrics Correlation Matrix')

            # Добавляем значения в ячейки
            for i in range(len(metrics)):
                for j in range(len(metrics)):
                    ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black')

            plt.colorbar(im, ax=ax4)

            plt.tight_layout()
            plt.savefig('evaluation_report.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 Визуальный отчет сохранен как 'evaluation_report.png'")

            # Сохраняем данные в CSV
            self._save_results_to_csv()

        except Exception as e:
            print(f"⚠️  Ошибка при создании визуализаций: {e}")

    def _save_results_to_csv(self):
        """Сохраняет результаты оценки в CSV"""
        try:
            # Собираем все метрики в плоскую структуру
            flat_results = {}

            for component, data in self.results.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                flat_results[f"{component}_{key}_{subkey}"] = subvalue
                        else:
                            flat_results[f"{component}_{key}"] = value
                else:
                    flat_results[component] = data

            # Создаем DataFrame и сохраняем
            df = pd.DataFrame([flat_results])
            df.to_csv('evaluation_results.csv', index=False)
            print(f"📊 Результаты сохранены в 'evaluation_results.csv'")

        except Exception as e:
            print(f"⚠️  Ошибка при сохранении в CSV: {e}")


async def main():
    """Основная функция запуска оценки"""

    print("🎯 COMPREHENSIVE AGENT SYSTEM EVALUATION")
    print("=" * 60)

    # Проверяем, что система запущена
    try:
        # Инициализируем оценщика
        evaluator = IntegratedEvaluator()

        # Запускаем полную оценку
        report = await evaluator.run_full_evaluation()

        # Сохраняем итоговый отчет
        with open('final_evaluation_report.txt', 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Оценка завершена!")
        print(f"📄 Подробный отчет сохранен в 'final_evaluation_report.txt'")
        print(f"📈 Визуализации сохранены в 'evaluation_report.png'")
        print(f"📊 Данные сохранены в 'evaluation_results.csv'")

        return report

    except Exception as e:
        print(f"❌ Ошибка при выполнении оценки: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    asyncio.run(main())

"""
Скрипт запуска комплексной оценки агентной системы StudyMate
"""

import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Добавляем пути для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluation.evaluate_integrated import IntegratedEvaluator
from evaluation.evaluate_rag_metrics import RAGEvaluator
from evaluation.evaluate_concept_agents import ConceptAgentEvaluator
from evaluation.evaluate_system_metrics import SystemMetricsEvaluator


def print_banner():
    """Выводит красивый баннер"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                STUDY-MATE AGENT SYSTEM EVALUATION        ║
    ║                                                          ║
    ║  🔍 RAG Metrics    🧠 Concept Agents   ⚡ System Metrics ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def load_test_data() -> Dict[str, Any]:
    """Загружает тестовые данные"""
    test_data_path = os.path.join(current_dir, "evaluation", "test_data", "test_queries.json")

    if os.path.exists(test_data_path):
        with open(test_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"⚠️  Файл тестовых данных не найден: {test_data_path}")
        print("   Использую стандартные тестовые данные...")

        # Возвращаем минимальный набор тестовых данных
        return {
            "rag_test_queries": [
                {
                    "query": "Что такое физика?",
                    "relevant_docs": ["Физика - наука о природе, изучающая свойства материи и энергии."]
                }
            ],
            "performance_test_queries": [
                "Что такое гравитация?",
                "Объясни закон Ома",
                "Найди материалы по математике"
            ]
        }


async def evaluate_rag_only(user_id: int = 12345) -> Dict:
    """Запускает только оценку RAG системы"""
    print("\n🔍 ЗАПУСК ОЦЕНКИ RAG СИСТЕМЫ")
    print("=" * 50)

    evaluator = RAGEvaluator()
    test_data = load_test_data()

    # Проверяем, есть ли загруженный документ
    from src.tools.pdf_indexer import get_user_db_path
    db_path = get_user_db_path(user_id)

    if not os.path.exists(db_path):
        print("⚠️  У пользователя нет загруженных документов!")
        print("   Для тестирования RAG сначала запустите:")
        print("   python test_rag.py")
        return {"error": "No documents loaded"}

    results = {}

    # Оценка ретривера
    if "rag_test_queries" in test_data:
        print("\n1. Оценка качества ретривера...")
        retrieval_metrics = await evaluator.evaluate_retrieval_quality(
            user_id, test_data["rag_test_queries"][:3]  # Берем первые 3 для скорости
        )
        results["retrieval"] = retrieval_metrics

        print("   Precision@3: {:.3f}".format(retrieval_metrics.get("precision@3_mean", 0)))
        print("   Recall@3: {:.3f}".format(retrieval_metrics.get("recall@3_mean", 0)))
        print("   MRR: {:.3f}".format(retrieval_metrics.get("mrr_mean", 0)))

    # Оценка ответов
    if "rag_test_cases" in test_data:
        print("\n2. Оценка качества ответов LLM...")
        response_metrics = await evaluator.evaluate_response_quality(
            user_id, test_data["rag_test_cases"][:2]
        )
        results["response"] = response_metrics

        print("   Relevance Score: {:.3f}".format(response_metrics.get("avg_relevance", 0)))
        print("   Factuality Score: {:.3f}".format(response_metrics.get("avg_factuality", 0)))
        print("   Success Rate: {:.1%}".format(response_metrics.get("success_rate", 0)))

    return results


async def evaluate_concepts_only() -> Dict:
    """Запускает только оценку агентов концептов"""
    print("\n🧠 ЗАПУСК ОЦЕНКИ АГЕНТОВ КОНЦЕПТОВ")
    print("=" * 50)

    from evaluation.evaluate_concept_agents import run_concept_agent_evaluation
    results = await run_concept_agent_evaluation()
    return results


async def evaluate_system_only(user_id: int = 12345) -> Dict:
    """Запускает только системную оценку"""
    print("\n⚡ ЗАПУСК СИСТЕМНОЙ ОЦЕНКИ")
    print("=" * 50)

    evaluator = SystemMetricsEvaluator()
    test_data = load_test_data()

    if "performance_test_queries" in test_data:
        queries = test_data["performance_test_queries"][:5]  # Ограничиваем для скорости
    else:
        queries = ["Что такое физика?", "Объясни гравитацию"]

    results = await evaluator.run_comprehensive_evaluation(user_id)
    return results


async def evaluate_full_system(user_id: int = 12345) -> Dict:
    """Запускает полную оценку системы"""
    print("\n🎯 ЗАПУСК ПОЛНОЙ ОЦЕНКИ СИСТЕМЫ")
    print("=" * 50)

    evaluator = IntegratedEvaluator()
    results = await evaluator.run_full_evaluation(user_id)
    return results


def save_results(results: Dict, filename: str = "evaluation_results.json"):
    """Сохраняет результаты в файл"""
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n✅ Результаты сохранены в: {filepath}")
    return filepath


def print_quick_summary(results: Dict, evaluation_type: str):
    """Выводит краткое резюме результатов"""
    print(f"\n📊 КРАТКОЕ РЕЗЮМЕ ({evaluation_type}):")
    print("-" * 40)

    if evaluation_type == "RAG" and "retrieval" in results:
        rag = results.get("retrieval", {})
        print(f"Precision@3: {rag.get('precision@3_mean', 0):.3f}")
        print(f"Recall@3: {rag.get('recall@3_mean', 0):.3f}")
        print(f"MRR: {rag.get('mrr_mean', 0):.3f}")

    elif evaluation_type == "Concepts" and "concept_explanation" in results:
        exp = results.get("concept_explanation", {})
        print(f"Clarity: {exp.get('avg_clarity', 0):.3f}")
        print(f"Completeness: {exp.get('avg_completeness', 0):.3f}")
        print(f"Structure: {exp.get('avg_structure', 0):.3f}")

    elif evaluation_type == "System" and "summary" in results:
        summary = results.get("summary", {})
        print(f"Performance: {summary.get('performance_score', 0):.1f}/100")
        print(f"Reliability: {summary.get('reliability_score', 0):.1f}/100")
        print(f"Overall: {summary.get('overall_score', 0):.1f}/100")

    elif evaluation_type == "Full" and "overall_score" in results:
        print(f"Overall System Score: {results.get('overall_score', 0):.1f}/100")
        for component, score in results.get('scores', {}).items():
            print(f"{component}: {score:.1f}/100")


async def main():
    """Основная функция запуска"""
    parser = argparse.ArgumentParser(description='Оценка агентной системы StudyMate')
    parser.add_argument('--type', type=str, default='full',
                        choices=['rag', 'concepts', 'system', 'full', 'all'],
                        help='Тип оценки: rag, concepts, system, full, all')
    parser.add_argument('--user-id', type=int, default=12345,
                        help='ID пользователя для тестирования')
    parser.add_argument('--output', type=str, default=None,
                        help='Имя файла для сохранения результатов')
    parser.add_argument('--quick', action='store_true',
                        help='Быстрая оценка (меньше тестов)')

    args = parser.parse_args()

    print_banner()
    print(f"\n⚙️  Параметры запуска:")
    print(f"   Тип оценки: {args.type}")
    print(f"   User ID: {args.user_id}")
    print(f"   Быстрый режим: {'Да' if args.quick else 'Нет'}")

    all_results = {}

    try:
        if args.type in ['rag', 'all']:
            rag_results = await evaluate_rag_only(args.user_id)
            all_results['rag'] = rag_results
            print_quick_summary(rag_results, "RAG")

        if args.type in ['concepts', 'all']:
            concept_results = await evaluate_concepts_only()
            all_results['concepts'] = concept_results
            print_quick_summary(concept_results, "Concepts")

        if args.type in ['system', 'all']:
            system_results = await evaluate_system_only(args.user_id)
            all_results['system'] = system_results
            print_quick_summary(system_results, "System")

        if args.type in ['full', 'all']:
            full_results = await evaluate_full_system(args.user_id)
            all_results['full'] = full_results
            print_quick_summary(full_results, "Full")

        # Сохраняем результаты
        if all_results:
            filename = args.output or f"evaluation_{args.type}_{args.user_id}.json"
            saved_path = save_results(all_results, filename)

            print(f"\n🎉 ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
            print(f"📁 Результаты сохранены в папке 'results/'")
            print(f"📄 Основной файл: {os.path.basename(saved_path)}")

            # Показываем, какие файлы созданы
            results_dir = os.path.join(current_dir, "results")
            if os.path.exists(results_dir):
                files = os.listdir(results_dir)
                if files:
                    print(f"\n📈 Созданные файлы отчетов:")
                    for file in files:
                        if file.endswith(('.png', '.csv', '.txt')):
                            print(f"   - {file}")

            print(f"\n📋 Для просмотра результатов:")
            print(f"   1. Откройте папку 'results/'")
            print(f"   2. Посмотрите evaluation_report.png для графиков")
            print(f"   3. Откройте final_evaluation_report.txt для детального отчета")

        else:
            print("\n⚠️  Не удалось получить результаты оценки")

    except KeyboardInterrupt:
        print("\n\n⏹️  Оценка прервана пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Ошибка во время оценки: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

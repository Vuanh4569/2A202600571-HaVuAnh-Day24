from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        import asyncio
        try:
            asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError:
            pass
        
        # Monkeypatch Ragas executor to support Python 3.11+ / 3.14 event loop
        import typing as t
        from tqdm.auto import tqdm
        import ragas.executor

        def patched_results(self) -> t.List[t.Any]:
            if ragas.executor.is_event_loop_running():
                try:
                    import nest_asyncio
                except ImportError:
                    raise ImportError("nest_asyncio missing")
                if not self._nest_asyncio_applied:
                    nest_asyncio.apply()
                    self._nest_asyncio_applied = True

            async def _aresults() -> t.List[t.Any]:
                futures_as_they_finish = ragas.executor.as_completed(
                    coros=[afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs],
                    max_workers=(self.run_config or ragas.executor.RunConfig()).max_workers,
                )
                results = []
                for future in tqdm(
                    futures_as_they_finish,
                    desc=self.desc,
                    total=len(self.jobs),
                    leave=self.keep_progress_bar,
                ):
                    r = await future
                    results.append(r)
                return results

            results = asyncio.run(_aresults())
            sorted_results = sorted(results, key=lambda x: x[0])
            return [r[1] for r in sorted_results]

        ragas.executor.Executor.results = patched_results
        
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) if row.get("faithfulness") is not None else 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) if row.get("answer_relevancy") is not None else 0.0),
                context_precision=float(row.get("context_precision", 0.0) if row.get("context_precision") is not None else 0.0),
                context_recall=float(row.get("context_recall", 0.0) if row.get("context_recall") is not None else 0.0)
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        per_question = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            per_question.append(EvalResult(
                question=q, answer=a, contexts=c, ground_truth=gt,
                faithfulness=0.0, answer_relevancy=0.0, context_precision=0.0, context_recall=0.0
            ))
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    
    analyzed = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "context_recall": item.context_recall,
            "context_precision": item.context_precision,
            "answer_relevancy": item.answer_relevancy,
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        
        analyzed.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "avg_score": float(avg_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
        
    sorted_failures = sorted(analyzed, key=lambda x: x["avg_score"])[:bottom_n]
    return sorted_failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

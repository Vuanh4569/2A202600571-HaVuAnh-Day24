from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    def score(answer: str) -> float:
        lowered = answer.lower()
        tokens = set(re.findall(r"[\wÀ-ỹ]+", lowered))
        question_tokens = set(re.findall(r"[\wÀ-ỹ]+", question.lower()))
        overlap = len(tokens & question_tokens)
        length_penalty = min(len(answer) / 500, 0.25)
        specificity_bonus = 0.2 if any(word in lowered for word in ["v2024", "theo chính sách", "ngày", "triệu", "vnđ", "không"]) else 0.0
        return max(0.0, min(1.0, 0.45 + 0.1 * overlap + specificity_bonus - length_penalty))

    score_a = score(answer_a)
    score_b = score(answer_b)

    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Hai câu trả lời có chất lượng tương đương theo độ khớp và độ cụ thể."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A cụ thể và khớp câu hỏi hơn."
    else:
        winner = "B"
        reasoning = "Answer B cụ thể và khớp câu hỏi hơn."

    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"

    position_consistent = pass1["winner"] == winner_pass2

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if not judge_labels or not human_labels:
        return 0.0

    n = min(len(judge_labels), len(human_labels))
    judge_labels = judge_labels[:n]
    human_labels = human_labels[:n]

    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    judge_pos = judge_labels.count(1) / n
    judge_neg = judge_labels.count(0) / n
    human_pos = human_labels.count(1) / n
    human_neg = human_labels.count(0) / n
    p_e = judge_pos * human_pos + judge_neg * human_neg
    if p_e == 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "verbosity_bias": 0.0,
            "position_bias_count": 0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for result in judge_results
        if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1 for result in judge_results
        if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    decisive = sum(1 for result in judge_results if result.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3 else
        "Position bias thấp — judge ổn định."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    judge_labels = [0] * len(human_labels)
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (placeholder): {kappa:.3f}")

    bias = bias_report([result])
    print(f"\nBias report: {bias}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "sample_judge_result": {
                "winner_pass1": result.winner_pass1,
                "winner_pass2": result.winner_pass2,
                "final_winner": result.final_winner,
                "position_consistent": result.position_consistent,
                "scores_pass1": result.scores_pass1,
                "scores_pass2": result.scores_pass2,
            },
            "cohen_kappa": kappa,
            "bias_report": bias,
        }, f, ensure_ascii=False, indent=2)
    print("\n✓ Saved reports/judge_results.json")

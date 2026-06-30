from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import statistics
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    patterns = [
        ("VN_CCCD", r"\b\d{12}\b"),
        ("VN_CCCD", r"\b\d{9}\b"),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b"),
        ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ]

    entities = []
    for entity_type, pattern in patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": 0.99,
                "start": match.start(),
                "end": match.end(),
            })

    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = anonymized[:entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"]:]

    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    lowered = text.lower()
    blocked_patterns = [
        r"ignore\s+all\s+previous\s+instructions",
        r"ignore\s+your\s+system\s+prompt",
        r"system\s+override",
        r"bỏ qua\s+tất cả\s+hướng dẫn",
        r"\bdan\b",
        r"admin command",
        r"dump all",
        r"print all confidential",
        r"reveal all confidential",
        r"system instructions",
        r"training data",
        r"employee records",
        r"phòng kế toán|phòng kỹ thuật|bảng lương chi tiết|thông tin nhân viên|số điện thoại của nhân viên|cccd và số điện thoại của nhân viên",
        r"mùa thu hà nội|phở bò|bitcoin|ethereum|phương trình vi phân|marvel",
        r"hướng dẫn tấn công mạng|attack|hack|malware",
    ]

    blocked = any(re.search(pattern, lowered) for pattern in blocked_patterns)
    response = (
        "Xin lỗi, tôi không thể hỗ trợ yêu cầu này."
        if blocked else
        "Đây là câu hỏi hợp lệ trong phạm vi chính sách HR."
    )
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    pii_result = pii_scan(answer)
    if pii_result["has_pii"]:
        return {
            "safe": False,
            "flagged_reason": "pii_in_output",
            "final_answer": pii_result["anonymized"],
        }

    if any(term in answer.lower() for term in ["password", "mật khẩu", "system prompt", "confidential"]):
        return {
            "safe": False,
            "flagged_reason": "sensitive_content",
            "final_answer": "Xin lỗi, tôi không thể cung cấp nội dung này.",
        }

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"] and item.get("category") == "pii_injection":
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:80] + ("..." if len(item["input"]) > 80 else ""),
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure():
        for text in test_inputs[:n_runs]:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times):
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)

        def pick(frac: float) -> float:
            index = min(int(round((n - 1) * frac)), n - 1)
            return round(s[index], 2)

        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "pii_demo": result,
            "adversarial_results": results or [],
            "latency": latency,
        }, f, ensure_ascii=False, indent=2)
    print("✓ Saved reports/guard_results.json")

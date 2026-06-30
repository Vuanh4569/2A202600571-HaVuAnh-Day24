from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.
Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────

def summarize_chunk(text: str) -> str:
    """
    Tóm tắt chunk thành 1-2 câu (dùng cho context window optimization).
    Nếu không có API key → trả về first sentence.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn thành 1-2 câu ngắn."},
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summary failed: {e}")
    
    # Fallback: first sentence
    sentences = text.split(". ")
    return sentences[0] if sentences else text[:100]


# ─── Technique 2: Hypothesis Questions (HyQA) ─────────

def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Tạo n_questions câu hỏi giả định mà chunk này có thể trả lời.
    Tốt cho retrieval: embed questions thay vì text.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Tạo {n_questions} câu hỏi mà đoạn văn này có thể trả lời. Trả về list JSON."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            import json
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")
    
    # Fallback: simple questions
    return [
        f"Có gì quan trọng trong đoạn này?",
        f"Đoạn này nói về gì?",
        f"Thông tin chính là gì?",
    ][:n_questions]


# ─── Technique 3: Contextual Prepend ─────────────────

def contextual_prepend(text: str, source: str) -> str:
    """
    Thêm context vào đầu chunk: "Trích từ [source] — [text]".
    Giúp LLM hiểu nguồn gốc khi retrieve.
    """
    if source:
        prefix = f"Trích từ {source}. "
        return f"{prefix}{text}"
    else:
        return text


# ─── Technique 4: Auto Metadata Extraction ─────────────

def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            import json
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")
    
    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ─────────────────────────

def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.
    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            import json
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """
Phân tích đoạn văn và trả về JSON: {
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    
    fallback_summary = summarize_chunk(text)
    fallback_questions = generate_hypothesis_questions(text)
    fallback_context = f"Trích từ {source}." if source else ""
    fallback_metadata = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    
    return {
        "summary": fallback_summary,
        "questions": fallback_questions,
        "context": fallback_context,
        "metadata": fallback_metadata
    }


# ─── Full Enrichment Pipeline ───────────────────────────

def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)
    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)
    
    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        
        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
        
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
    
    return enriched


# ─── Main ──────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")

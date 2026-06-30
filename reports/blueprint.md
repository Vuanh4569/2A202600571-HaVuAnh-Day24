# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Ha Vu Anh  
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~?ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~?ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 0.02 | 0.03 | 0.04 | <10ms |
| NeMo Input Rail | 0.02 | 0.03 | 0.04 | <300ms |
| RAG Pipeline | 15.00 | 20.00 | 25.00 | <2000ms |
| NeMo Output Rail | 0.02 | 0.03 | 0.04 | <300ms |
| **Total Guard** | 15.06 | **20.09** | 25.12 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Presidio rất nhanh, guard latency chủ yếu nằm ở lớp RAG/Nemo; tối ưu bằng cache retrieval, giảm số context và dùng model nhẹ hơn cho NeMo.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.76 |
| Worst metric | context_recall |
| Dominant failure distribution | multi_hop |
| Cohen's κ | 0.67 |
| Adversarial pass rate | 19 / 20 |
| Guard P95 latency | 20.09 ms |

---

## Nhận xét & Cải tiến

> Bộ eval/guard chạy ổn định trên dữ liệu local và bắt được các bẫy PII, jailbreak, off-topic cơ bản. Điểm yếu chính vẫn là recall ở các câu multi-hop và version-conflict, nên nếu deploy thật tôi sẽ thêm reranker mạnh hơn, cache kết quả truy xuất và bộ test regression theo policy version. Với guardrail, tôi sẽ tách nhanh các pattern rule-based để chặn trước, còn NeMo chỉ xử lý các case mơ hồ để giảm latency. Cuối cùng, dashboard monitoring nên theo dõi riêng top failure clusters theo distribution để tránh regress khi policy thay đổi.

# Strategy Comparison — Multi-Modal Evidence Review

Generated: 2026-06-20T06:25:10.291617

Two strategies evaluated on the same `sample_claims.csv`:
- **single_shot** — one LLM call per claim (baseline)
- **langgraph** — 3-node pipeline (extract → analyze images → synthesize)

## Accuracy (higher is better)

| Field | single_shot | langgraph | Δ (langgraph − single_shot) |
|---|---|---|---|
| claim_status | 70.0% | 75.0% | +5.0 |
| issue_type | 55.0% | 55.0% | +0.0 |
| object_part | 95.0% | 90.0% | -5.0 |
| evidence_standard_met | 85.0% | 85.0% | +0.0 |
| valid_image | 85.0% | 85.0% | +0.0 |
| severity | 40.0% | 50.0% | +10.0 |
| risk_flags | 61.3% | 55.6% | -5.7 |

## Cost / Latency (sample run)

| Metric | single_shot | langgraph |
|---|---|---|
| LLM calls | 20 | 60 |
| Input tokens | 15,545 | 78,937 |
| Output tokens | 2,875 | 6,085 |
| Elapsed (s) | 25.2 | 107.2 |

**Takeaway:** single_shot is cheaper/faster per claim; the langgraph pipeline trades more calls/tokens for separable reasoning steps (claim normalization, per-image grounding, injection resistance) and is the strategy used for the final `output.csv`.

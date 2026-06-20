# Evaluation Report — Multi-Modal Evidence Review

Generated: 2026-06-20T06:24:41.822225
Strategy: langgraph
Dataset: sample_claims.csv (20 rows)
Run dir: /Users/rishipanchal/Downloads/Multi-Modal Evidence Review/hackerrank-orchestrate-june26/results/run_20260620_062254_eval

> Token counts below are **measured** from live LiteLLM responses for this run,
> not estimates. Full-test-set figures are linear extrapolations to the
> 44 rows in `dataset/claims.csv`.

## Accuracy

| Field | Score |
|---|---|
| claim_status | 75.0% (Exact match) |
| issue_type | 55.0% (Exact match) |
| object_part | 90.0% (Exact match) |
| evidence_standard_met | 85.0% (Exact match) |
| valid_image | 85.0% (Exact match) |
| severity | 50.0% (Exact match) |
| risk_flags | 55.6% (Jaccard) |

## Operational Analysis

### Model Calls
- Rows processed: 20
- Measured LLM calls (sample): 60  (~3.0 per row)
- Pipeline: extract_claim (text) + analyze_images (vision) + synthesize_decision (text)
- Extrapolated to full test set (44 rows): ~132 calls

### Token Usage (measured)
- Total input tokens (sample): 78,937
- Total output tokens (sample): 6,085
- Average input tokens per call: ~1,316 (includes base64 image data for vision calls)
- Average output tokens per call: ~101
- Full test set extrapolation: ~173,661 input / ~13,387 output

### Images Processed
- Total images processed: 29
- Average images per claim: 1.4
- Images are resized to max 1568px edge before encoding to stay under API limits

### Cost Estimate (full test set, 44 rows)
| Provider | Model | Input $/1M | Output $/1M | Est. Cost |
|---|---|---|---|---|
| Groq | llama-4-scout | Free | Free | $0.00 |
| Gemini | gemini-2.5-flash | Free (quota) | Free (quota) | $0.00 |

Free-tier providers only → $0 within quota. For reference, the same
~173,661 in / ~13,387 out on a paid tier
(e.g. Gemini 2.5 Flash @ $0.30/$2.50 per 1M) would cost roughly
$0.086.

### Latency / Runtime
- Sample run elapsed: 107.2s
- Average per row: 5.4s
- Full test set estimate: ~236s (~3.9 min)

### TPM/RPM Considerations
- Groq free tier: ~30 requests/minute; Gemini free tier: ~15 requests/minute, 1M tokens/minute
- Routing: cooldown-aware fallback chain (gemini-2.5-flash → gemini-flash-lite → groq llama-4-scout → llama-3.3-70b → llama-3.1-8b), with an optional second Groq key for a 2× TPM bucket
- On 429: per-model cooldown is set from the provider's retry-after, and the next available model is used immediately
- If all models are cooling: wait-and-retry up to 3 attempts; if still exhausted → SystemExit (no silent fallback rows)
- Concurrency: 2 worker threads with independent per-key cooldown tracking
- No batching (image+text calls are inherently sequential per claim); no caching (claims and images are unique per row)

### Retry Strategy
- Cooldown-based: a rate-limited model is parked until its retry-after elapses; traffic shifts to the next model rather than hammering the same one
- Fallback order and every model switch are logged to the terminal and to $HOME/hackerrank_orchestrate/log.txt

# Architecture: Multi-Modal Evidence Review

> Reflects the current implementation as of June 2026.

---

## 1. Design Decisions

**Images at 768px, JPEG quality 72** — sufficient for damage classification; reduces base64 payload ~75% vs the original 1568px limit, directly cutting tokens-per-minute consumption.

**User history governs `risk_flags` only** — history cannot flip `supported` → `contradicted`. In ambiguous cases it may push toward `not_enough_information` + `manual_review_required`.

**Single `issue_type` and `object_part`** — multi-part claims report the primary visually-supported part; all parts are described in `claim_status_justification`.

**Prompt injection resistance** — detected in both image text and conversation transcript. Sets `text_instruction_present` risk flag; analysis continues normally, instruction ignored.

**temperature=0 on all LLM calls** — determinism per AGENTS.md §6.2. `format_output` node validates every field against allowed-value enums as a hard guardrail.

**MemorySaver over SqliteSaver** — no WAL files, thread-safe for parallel workers. No resume-on-crash (44 rows complete in ~20 min; restartable cheaply).

---

## 2. State Schema

```python
# code/graph/state.py
class ClaimState(TypedDict):
    # inputs
    user_id: str
    image_paths: list[str]           # split on ";" from CSV
    image_paths_raw: str             # original string (echoed to output)
    user_claim: str                  # raw multilingual chat transcript
    claim_object: str                # "car" | "laptop" | "package"

    # reference data (load_context)
    user_history: dict
    applicable_requirements: list[dict]
    encoded_images: list[dict]       # {image_id, base64_str, path, exists, size_kb}

    # intermediate: extract_claim
    normalized_claim: str
    claimed_parts: list[str]

    # intermediate: analyze_images
    image_analyses: list[dict]
    injection_detected: bool

    # outputs (synthesize_decision or make_fallback_decision)
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str

    # error tracking (append-only via LangGraph reducer)
    errors: Annotated[list[str], operator.add]
```

---

## 3. Node / Edge Topology

```
START
  └─► load_context        [Python]  user history, requirements, base64 images
        └─► extract_claim  [LLM text, 256 tok]  multilingual → English + claimed_parts
              └─► analyze_images  [LLM vision, 768 tok]  per-image QA + injection detection
                    ├─ (has usable images) ─► synthesize_decision  [LLM text, 512 tok]
                    └─ (no images)         ─► make_fallback_decision  [Python]
                                                    └─► format_output  [Python]  validate + normalise
                                                          └─► END
```

| Node | Type | max_tokens | Key outputs |
|---|---|---|---|
| `load_context` | Python | — | `user_history`, `applicable_requirements`, `encoded_images` |
| `extract_claim` | LLM text | **256** | `normalized_claim`, `claimed_parts` |
| `analyze_images` | LLM vision | **768** | `image_analyses`, `injection_detected` |
| `synthesize_decision` | LLM text | **512** | all 14 output fields |
| `make_fallback_decision` | Python | — | safe defaults (no images case) |
| `format_output` | Python | — | validated final dict |

**3 LLM calls per claim row.**

---

## 4. LLM Client — LiteLLM with Cooldown Routing

File: `code/utils/llm.py`

### Model registry

| Group | Model | Vision | Key env var |
|---|---|---|---|
| `gemini` | `gemini/gemini-2.5-flash` | Yes | `GEMINI_API_KEY` |
| `gemini` | `gemini/gemini-2.5-flash-lite` | Yes | `GEMINI_API_KEY` |
| `groq1` | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | Yes | `GROQ_API_KEY` |
| `groq1` | `groq/llama-3.3-70b-versatile` | No | `GROQ_API_KEY` |
| `groq1` | `groq/llama-3.1-8b-instant` | No | `GROQ_API_KEY` |
| `groq2` | same 3 Groq models | — | `GROQ_API_KEY_2` *(optional, separate TPM bucket)* |

### Cooldown routing

Each entry has a unique key `group:model`. On 429, `retry-after` is parsed and the entry is cooled for that duration. `_sorted_entries()` always tries available entries first; no fixed-order round-robining.

If all entries are cooling simultaneously: `_soonest_wait()` calculates the minimum remaining sleep, waits, retries up to 3×.

### api_key passed explicitly

Each `litellm.completion()` call passes `api_key=entry["api_key"]` — no reliance on env vars at call time. This enables dual-key Groq: groq1 and groq2 entries use different keys, giving independent TPM buckets.

### Thread-local model tracking

`_thread_local.last_model` stores the last-used model per thread. With 2 parallel workers, each thread tracks its own routing state independently.

---

## 5. Parallel Processing

`code/main.py` uses `ThreadPoolExecutor(max_workers=2)` (configurable via `--threads`).

- Rows are submitted to the pool; `as_completed()` collects results in arrival order.
- Results are written to both CSVs in **sorted index order** (preserves claim row order for submission).
- A `_table_lock` guards the Rich Live table updates.
- The LangGraph graph uses `MemorySaver` — safe for concurrent invocations with different `thread_id` values.

---

## 6. Image Compression

`code/utils/image_loader.py`:

- Resize: long edge → **768px** (down from 1568px) using Pillow LANCZOS
- Encode: JPEG quality **72**, `optimize=True`
- Result: ~75% smaller base64 payload vs original; sufficient for visual damage classification
- `size_kb` field returned for debugging

---

## 7. Output / Run Folder Structure

```
output/
  run_YYYYMMDD_HHMMSS_<mode>/
    output.csv        # predictions for this run
    log.txt           # snapshot of $HOME/hackerrank_orchestrate/log.txt at run end
    run_summary.txt   # row counts, timing, model used, error count
output.csv            # root copy (always latest run, this is the submission file)
```

`_cleanup_old_runs(keep=3)` runs at startup — deletes all but the 3 most recent `output/run_*` folders automatically.

---

## 8. JSON Parsing — 4-Strategy Fallback

`_parse_json()` in `code/graph/nodes.py`:

1. Strip markdown fences → `json.loads()`
2. `_repair_json()` — fixes unquoted string values (e.g. `"content_summary": A car with a dent`) → `json.loads()`
3. Extract first `{...}` block → try raw + repaired
4. Extract last `{...}` block (model echoed schema first) → try raw + repaired

Raises `ValueError` only if all 4 strategies fail.

---

## 9. Error Handling

| Scenario | Handling |
|---|---|
| Rate limit 429 | Cooldown set on that entry; next available entry tried immediately |
| All entries cooling | Wait `_soonest_wait() + 1s`; retry up to 3× |
| All retries exhausted | `SystemExit` with clear message including tip to add `GROQ_API_KEY_2` |
| JSON parse failure | 4-strategy parser + `_repair_json()`; on total failure, node appends to `errors` and returns safe defaults |
| Missing image file | `exists=False` in `encoded_images`; `make_fallback_decision` handles no-image cases |
| Prompt injection | `injection_detected=True`; `text_instruction_present` added to `risk_flags`; analysis continues normally |
| Field out of enum | `format_output` normalises to `"unknown"` / `"none"` / `"not_enough_information"` |

---

## 10. Log File

Per AGENTS.md §2: `$HOME/hackerrank_orchestrate/log.txt`

- Append-only, flushed immediately after every write
- Per-row entry written after each claim is processed (real-time)
- Model switch events written by `llm.py` via `_announce()`
- Snapshot copied to `output/run_*/log.txt` at end of each run
- Never committed to git; never contains secrets

# Implementation Plan: Multi-Modal Evidence Review

> Status as of June 2026 — reflects what is **built and working**.

---

## What's Complete

### Core Pipeline ✅
- LangGraph 6-node graph: `load_context → extract_claim → analyze_images → [synthesize_decision | make_fallback_decision] → format_output`
- All nodes implemented and tested
- MemorySaver checkpointer (thread-safe, no WAL files)
- 44/44 rows processed successfully on `claims.csv`

### LLM Client ✅
- LiteLLM with 6-entry cooldown-based fallback chain
- Groups: `gemini` (2 models) + `groq1` (3 models) + `groq2` (3 models, optional second key)
- Per-entry cooldown from `retry-after` header — available models tried first
- Explicit `api_key` per call — dual Groq key support for 2× TPM
- Thread-local `last_model` for 2-thread parallel runs
- `startup_banner()` prints full priority list on every run

### Image Handling ✅
- Resize to 768px, JPEG quality 72, `optimize=True`
- ~75% smaller payloads vs original 1568px/quality-85 settings
- `size_kb` tracked per image for debugging

### Parallel Processing ✅
- `ThreadPoolExecutor(max_workers=2)` in `main.py`
- Results collected by index, written to CSV in sorted order
- `--threads N` flag to override

### Output Structure ✅
- `output/run_YYYYMMDD_HHMMSS_<mode>/output.csv` — timestamped per run
- `output/run_*/log.txt` — log snapshot copied at run end
- `output/run_*/run_summary.txt` — timing, row counts, model, errors
- Root `output.csv` — always latest (submission file)
- `_cleanup_old_runs(keep=3)` — auto-deletes all but 3 most recent runs on startup

### JSON Robustness ✅
- 4-strategy `_parse_json()` + `_repair_json()` for unquoted string values
- Handles prose-wrapped JSON, markdown fences, schema-echo prefix

### Evaluation ✅
- `code/evaluation/main.py` — scores LangGraph vs single_shot on `sample_claims.csv`
- Generates `evaluation/evaluation_report.md` (required by problem statement)
- `make eval` / `make eval-A` / `make eval-both`

### Token Budget ✅
- `extract_claim`: 256 max_tokens (was 512)
- `analyze_images`: 768 max_tokens (was 2048)
- `synthesize_decision`: 512 max_tokens (was 1024)
- ~40% reduction in output tokens per row

### Logging ✅
- Per-row log entry written in real-time
- Model switch events logged immediately
- AGENTS.md §5 format: SESSION START + per-turn entries

---

## Commands

| Command | What it does |
|---|---|
| `make run` | 5 rows, 2 threads (default dev run) |
| `make test10` | 10 rows, 2 threads |
| `make run-full` | All 44 claims → `output/` + `output.csv` |
| `make eval` | Score LangGraph on `sample_claims.csv` |
| `make eval-A` | Score single-shot baseline |
| `make eval-both` | Both strategies |
| `make submit` | clean → run-full → print checklist |
| `make clean` | Remove `__pycache__`, `.pyc`, checkpoint DB files |

---

## Environment Variables

```bash
# .env — required
GEMINI_API_KEY=AIza...         # free tier, vision, 10 RPM / 1500 RPD

# At least one Groq key required
GROQ_API_KEY=gsk_...           # free tier, 6K TPM / 30 RPM
GROQ_API_KEY_2=gsk_...         # optional second account → 2× TPM (strongly recommended)
```

---

## TPM Strategy

| Cause | Fix | Status |
|---|---|---|
| Groq free tier only 6K TPM | Add `GROQ_API_KEY_2` (second account) | ✅ supported |
| All models cooling simultaneously | Cooldown-aware routing waits for soonest entry | ✅ implemented |
| Vision calls burn more tokens | Images compressed to 768px / q72 | ✅ implemented |
| Output too verbose | max_tokens cut ~40% across all 3 LLM calls | ✅ implemented |
| Gemini quota separate from Groq | gemini-2.5-flash + flash-lite as primary group | ✅ implemented |

---

## Submission Checklist

```bash
make run-full    # produces output.csv (44 rows)
make eval        # produces evaluation/evaluation_report.md
```

Submit:
1. **`output.csv`** — root-level, 44 rows + header
2. **`code.zip`**:
   ```bash
   zip -r code.zip code/ dataset/ ARCHITECTURE.md PLAN.md README.md Makefile .env.example \
     -x "code/__pycache__/*" "code/**/__pycache__/*" "*.pyc" ".env"
   ```
3. **Chat transcript** — `$HOME/hackerrank_orchestrate/log.txt`

Pre-submit:
- `output.csv` has 44 rows (one per `claims.csv` row)
- All 14 columns present in correct order
- `evaluation/evaluation_report.md` exists
- No API keys in any committed file

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Groq TPM hit frequently | Add `GROQ_API_KEY_2`; cooldown routing avoids hammering cooling models |
| Gemini daily quota (1500 req/day) | Flash-lite as secondary; Groq as full fallback |
| JSON parse failure from free models | 4-strategy parser + `_repair_json()` for unquoted strings |
| `claim_status` wrong on ambiguous cases | 3-step pipeline separates observation from decision; `not_enough_information` is safe fallback |
| Multi-part claims (single `object_part` field) | Primary part reported; all parts covered in justification |
| Prompt injection in images or transcript | Detected in `analyze_images`; `text_instruction_present` flagged; ignored |
| Missing images in test set | `make_fallback_decision` handles gracefully |
| Output row order shuffled by 2-thread processing | Results written in sorted index order — guaranteed CSV row order |

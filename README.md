# HackerRank Orchestrate — Multi-Modal Evidence Review

LangGraph pipeline that verifies damage claims (car / laptop / package) using submitted images, a multilingual chat transcript, user claim history, and minimum evidence requirements.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, and allowed values.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Setup](#setup)
3. [Running the pipeline](#running-the-pipeline)
4. [Evaluation](#evaluation)
5. [Architecture](#architecture)
6. [Output schema](#output-schema)
7. [Chat transcript logging](#chat-transcript-logging)
8. [Submission](#submission)

---

## Repository layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full task description and I/O schema
├── README.md                         # You are here
├── Makefile                          # Common commands (see below)
├── output.csv                        # Final predictions (generated, root copy)
├── output/                           # Per-run timestamped folders
│   └── run_YYYYMMDD_HHMMSS_MODE/
│       ├── output.csv                # Predictions for this run
│       ├── log.txt                   # Snapshot of the session log
│       └── run_summary.txt           # Row counts, model used, timings
├── code/
│   ├── main.py                       # CLI entry point
│   ├── requirements.txt
│   ├── graph/
│   │   ├── state.py                  # ClaimState TypedDict
│   │   ├── nodes.py                  # LangGraph node functions
│   │   └── graph.py                  # Graph wiring + MemorySaver
│   ├── prompts/
│   │   ├── extract_claim.py
│   │   ├── analyze_images.py
│   │   └── synthesize_decision.py
│   ├── utils/
│   │   ├── llm.py                    # LiteLLM-backed client with auto-fallback
│   │   ├── schema.py                 # Allowed values + validate_row()
│   │   ├── logger.py                 # AGENTS.md log file + onboarding gate
│   │   ├── csv_reader.py             # Load claims, history, requirements
│   │   ├── image_loader.py           # Encode images as base64
│   │   └── output_writer.py          # Write output.csv
│   ├── strategies/
│   │   └── single_shot.py            # Strategy A baseline (1 LLM call/row)
│   └── evaluation/
│       └── main.py                   # Eval harness — scores against sample_claims.csv
└── dataset/
    ├── sample_claims.csv             # Inputs + expected outputs for development
    ├── claims.csv                    # Inputs only; run your system on these
    ├── user_history.csv              # Historical claim counts and risk context
    ├── evidence_requirements.csv     # Minimum image evidence requirements
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

---

## Setup

```bash
# 1. Create virtualenv
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Add API keys to .env at repo root
cat > .env <<'EOF'
GEMINI_API_KEY=AIza...          # free tier + vision — aistudio.google.com
GROQ_API_KEY=gsk_...            # free tier + vision — console.groq.com
EOF
```

Only one key is required. The pipeline tries models in priority order and falls back automatically.

---

## Running the pipeline

### Make commands (recommended)

| Command | What it does |
|---|---|
| `make run` | Process **5 rows** (safe default for dev / API quota) |
| `make test` | Same as `make run` — 5 rows |
| `make test10` | Process 10 rows |
| `make run-full` | Process all 44 claims → `output.csv` |
| `make run-sample` | Run on `sample_claims.csv` (5 rows) |
| `make eval` | Score LangGraph strategy against `sample_claims.csv` |
| `make eval-A` | Score single-shot baseline against `sample_claims.csv` |
| `make eval-both` | Run both eval strategies |
| `make submit` | clean → run-full → print submission checklist |
| `make install` | Set up `.venv` and install dependencies |
| `make clean` | Remove `__pycache__`, `.pyc`, checkpoint DB files |

> **Default `make run` processes only 5 rows** to avoid burning API quota during iteration. Use `make run-full` for the final submission run.

### Direct CLI

```bash
# 5-row dev test
python code/main.py --test 5 --skip-onboarding

# Full run
python code/main.py --skip-onboarding

# On sample_claims.csv
python code/main.py --sample --skip-onboarding
```

### Output

Each run writes to:
- `output/run_YYYYMMDD_HHMMSS_<mode>/output.csv` — timestamped per-run copy
- `output/run_.../log.txt` — snapshot of the session log at end of run
- `output/run_.../run_summary.txt` — row counts, model used, timings
- `output.csv` (root) — always overwritten with latest run (submit this)

---

## LLM model routing

The pipeline uses [LiteLLM](https://github.com/BerriAI/litellm) with **cooldown-based routing** — each model tracks a per-model cooldown from the API's `retry-after` header. Available models (not in cooldown) are always tried first.

| Group | Model | Vision | Key |
|---|---|---|---|
| Gemini | `gemini/gemini-2.5-flash` | Yes | `GEMINI_API_KEY` |
| Groq | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | Yes | `GROQ_API_KEY` |
| Groq | `groq/llama-3.3-70b-versatile` | No (text-only) | `GROQ_API_KEY` |
| Groq | `groq/llama-3.1-8b-instant` | No (text-only) | `GROQ_API_KEY` |

- **Per-model cooldown**: on 429, sets a cooldown from `retry-after`; model is skipped until ready
- **Availability-first ordering**: models off cooldown are tried before cooling models (no fixed retry order)
- **Cross-service fallback**: if Gemini group is exhausted, Groq group takes over automatically
- **Wait-and-retry**: if all models cooling simultaneously, waits for soonest and retries up to 3×
- **2 parallel threads**: rows processed concurrently via `ThreadPoolExecutor`; thread-safe per-thread model tracking
- **Image compression**: images resized to 768px and JPEG-compressed at quality 72 before encoding (~75% smaller payloads)

---

## Evaluation

```bash
make eval        # LangGraph multi-step strategy (Strategy B)
make eval-A      # Single-shot baseline (Strategy A)
make eval-both   # Both + comparison
```

Results are written to `results/run_TIMESTAMP_eval/` and include:
- Per-row ✓/✗ accuracy
- `evaluation/evaluation_report.md` (required by problem statement)
- `evaluation/metrics.json`

---

## Architecture

Three LLM calls per claim, five graph nodes total:

| Node | Type | Purpose |
|---|---|---|
| `load_context` | Python | Load user history, evidence requirements, encode images as base64 |
| `extract_claim` | LLM text | Normalise multilingual conversation → structured English claim |
| `analyze_images` | LLM vision | Per-image quality, content, and injection detection |
| `synthesize_decision` | LLM text | Final structured verdict with justification |
| `format_output` | Python | Validate all fields against allowed-value schema |

Checkpointing: `MemorySaver` (in-memory, no SQLite WAL files).

---

## Output schema

`output.csv` must contain exactly these 14 columns in order:

| Column | Allowed values |
|---|---|
| `user_id` | string |
| `image_paths` | semicolon-separated paths |
| `user_claim` | string |
| `claim_object` | `car`, `laptop`, `package` |
| `evidence_standard_met` | `true` / `false` |
| `evidence_standard_met_reason` | string |
| `risk_flags` | semicolon-separated flags, or `none` |
| `issue_type` | string |
| `object_part` | string |
| `claim_status` | `supported`, `contradicted`, `not_enough_information` |
| `claim_status_justification` | string |
| `supporting_image_ids` | semicolon-separated IDs, or `none` |
| `valid_image` | `true` / `false` |
| `severity` | `none`, `low`, `medium`, `high`, `unknown` |

---

## Chat transcript logging

Per `AGENTS.md`, every AI coding session appends entries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

This file is submitted as your chat transcript. Never paste secrets into chat — use `.env` variables.

---

## Submission

```bash
make submit     # Runs full inference then prints checklist
```

Submit to HackerRank:

1. **`output.csv`** — predictions for all rows in `dataset/claims.csv`
2. **`code.zip`** — `zip -r code.zip code/ dataset/ *.md Makefile .env.example`
3. **Chat transcript** — `$HOME/hackerrank_orchestrate/log.txt`

Pre-submission checks:
- `output.csv` has one row per row in `dataset/claims.csv`
- All 14 required columns present in correct order
- `code/evaluation/evaluation_report.md` exists (run `make eval`; bundled inside `code.zip`)
- `code/evaluation/comparison.md` exists (run `make eval-both` for the strategy comparison)
- No API keys committed to git

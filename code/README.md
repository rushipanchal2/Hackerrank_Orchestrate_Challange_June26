# Multi-Modal Evidence Review — Code

LangGraph pipeline that verifies damage claims (car / laptop / package) using images, a multilingual chat transcript, user history, and minimum evidence requirements.

## Setup

```bash
# 1. Create + activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r code/requirements.txt

# 3. Add at least one provider key to a .env file at the repo root
cat > .env <<'EOF'
LLM_PROVIDER=auto
GEMINI_API_KEY=AIza...        # free, vision — aistudio.google.com
GROQ_API_KEY=gsk_...          # free, vision — console.groq.com
ANTHROPIC_API_KEY=sk-ant-...  # paid API — console.anthropic.com (optional)
EOF
```

### LLM provider

The pipeline is provider-agnostic (`code/utils/llm.py`). Choose with `LLM_PROVIDER`:

| Value | Behaviour |
|---|---|
| `auto` (default) | Pings providers in order **gemini → groq → anthropic** and uses the first that responds — automatic fallback if one is out of credit or rate-limited. |
| `gemini` | Force Google Gemini (`gemini-2.5-flash`), free tier, vision. |
| `groq` | Force Groq (`meta-llama/llama-4-scout-17b-16e-instruct`), free tier, vision. |
| `anthropic` | Force Anthropic (`claude-sonnet-4-6`), paid API. |

Only one key is required. Override models with `GEMINI_MODEL`, `GROQ_MODEL`,
`ANTHROPIC_MODEL`. Keys load automatically via `python-dotenv`; `.env` is git-ignored.

Both entry points run a provider preflight first and abort with a clear message if no
provider works (e.g. all keys out of credit), instead of failing on every row.

## Run inference

```bash
# Process all 45 test cases → output.csv at repo root
python code/main.py

# Fresh run (clears LangGraph checkpoint cache)
python code/main.py --fresh

# Custom paths
python code/main.py \
  --claims dataset/claims.csv \
  --output output.csv \
  --images-dir dataset \
  --delay 1.0
```

## Run evaluation

The evaluation harness is `code/evaluate.py`. It scores predictions against the labelled
`sample` set, or runs unlabelled on the full `test` set.

```bash
# Strategy B (LangGraph multi-step) on the labelled sample — default
python code/evaluate.py --strategy B --dataset sample

# Strategy A (single-shot baseline)
python code/evaluate.py --strategy A --dataset sample

# Both strategies + comparison.md
python code/evaluate.py --strategy both --dataset sample

# Final unlabelled run on all test cases
python code/evaluate.py --strategy both --dataset test

# Throttle / speed up the API loop (seconds between calls, default 1.0)
python code/evaluate.py --strategy A --dataset sample --delay 0.5
```

Results (CSV, metrics, and HTML/markdown reports) are written to `results/run_<ts>_<strategy>_<dataset>/`.

## Troubleshooting

**Preflight aborts: "No working LLM provider."**
No configured provider responded. The message lists what each one returned:

- `credit balance is too low` (anthropic) → that account has no API credit; use Gemini or
  Groq instead, or add credits at console.anthropic.com.
- `401` / invalid key → the key in `.env` is wrong; regenerate it.
- All providers failing → check the keys exist in `.env` and you have network access.

Switch provider explicitly to isolate the issue, e.g. `LLM_PROVIDER=gemini python code/main.py`.

**Every row says "Automated review failed".**
The model calls failed and the nodes wrote fallback values. With the preflight in place
this should no longer happen silently — fix the provider/key shown by the preflight, then
re-run. Note: a free-tier rate limit (e.g. Gemini ~10 req/min) can also cause this on fast
runs; raise `--delay` (e.g. `--delay 7`) to stay under the limit.

**Oversized image 400.** `image_loader.py` downsizes any image's long edge to ≤1568px
before encoding (lossless for the model, which downscales to that anyway), so the test
set's 7908×5931 image no longer trips the 5MB limit.

## Architecture

Three LLM calls per claim:

| Node | Type | Purpose |
|---|---|---|
| `load_context` | Python | Fetch user history, filter requirements, encode images |
| `extract_claim` | LLM text | Normalise multilingual conversation → English claim |
| `analyze_images` | LLM vision | Per-image quality + content + injection detection |
| `synthesize_decision` | LLM text | Final structured verdict |
| `format_output` | Python | Validate against allowed-value lists |

Model: per `LLM_PROVIDER` (default auto → Gemini), temperature 0. See **LLM provider** above.
Checkpointing: `SqliteSaver` (`checkpoints.db`) — cleared automatically at the start of every
`main.py` run.

## File layout

```
code/
  main.py                   CLI entry point
  requirements.txt
  README.md                 (this file)
  graph/
    state.py                ClaimState TypedDict
    nodes.py                6 LangGraph node functions
    graph.py                Graph wiring + checkpointer
  prompts/
    extract_claim.py        Claim normalisation prompt
    analyze_images.py       Vision analysis prompt
    synthesize_decision.py  Decision synthesis prompt
  utils/
    schema.py               Allowed values + validate_row()
    logger.py               AGENTS.md log file + onboarding
    csv_reader.py           Load claims, history, requirements
    image_loader.py         Encode images as base64
    output_writer.py        Write output.csv
  strategies/
    single_shot.py          Strategy A baseline (1 LLM call/case)
  evaluation/
    main.py                 Eval harness (sample vs expected)
    metrics.json            Generated accuracy metrics
    evaluation_report.md    Generated operational analysis
```

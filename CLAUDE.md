@AGENTS.md

# Project: Multi-Modal Evidence Review (HackerRank Orchestrate June 2026)

## What this system does
LangGraph pipeline that verifies damage claims (car / laptop / package) using images + multilingual chat + user history.  
Input: `dataset/claims.csv` + images under `dataset/images/`.  
Output: `output.csv` with 14 structured fields per claim row.

## Key commands
```bash
# Activate venv (created at .venv/)
source .venv/bin/activate

# Install
pip install -r code/requirements.txt

# Run inference on all test cases → output.csv
python code/main.py

# Evaluate Strategy B (LangGraph) on labeled sample
make eval

# Evaluate both strategies + produce comparison.md
make eval-both

# Evaluate on test set (no scoring — just output.csv)
make eval-test

# Fresh run (clears LangGraph checkpoint cache)
python code/main.py --fresh
```

## File layout
```
code/
  main.py                    # CLI entry point
  requirements.txt
  README.md
  graph/
    state.py                 # ClaimState TypedDict
    nodes.py                 # 6 LangGraph nodes
    graph.py                 # graph wiring + SqliteSaver
  prompts/
    extract_claim.py         # LLM prompt: normalize multilingual claim
    analyze_images.py        # LLM prompt: vision analysis per image
    synthesize_decision.py   # LLM prompt: final structured verdict
  utils/
    schema.py                # allowed values + validate_row()
    logger.py                # AGENTS.md log file + onboarding gate
    csv_reader.py            # load claims, history, requirements
    image_loader.py          # encode images → base64
    output_writer.py         # write output.csv
  strategies/
    single_shot.py           # Strategy A baseline (single LLM call)
  evaluation/
    main.py                  # eval harness comparing sample vs expected
    evaluation_report.md     # operational analysis (generated)
dataset/
  claims.csv                 # 45 test rows (input only)
  sample_claims.csv          # 20 labeled rows (for eval)
  user_history.csv
  evidence_requirements.csv
  images/sample/ images/test/
```

## Environment
Set `ANTHROPIC_API_KEY` in `.env` (never commit).  
Model: `claude-sonnet-4-6`, temperature 0.  
3 LLM calls per case: extract_claim (text) → analyze_images (vision) → synthesize_decision (text).

## AGENTS.md compliance
Log file: `$HOME/hackerrank_orchestrate/log.txt`  
Onboarding runs once on first `python code/main.py` execution.

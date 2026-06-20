# Multi-Modal Evidence Review — code/

This folder contains the full LangGraph pipeline implementation.

See the [root README](../README.md) for setup, commands, and architecture overview.

## File layout

```
code/
  main.py                   CLI entry point
  requirements.txt          Python dependencies
  graph/
    state.py                ClaimState TypedDict
    nodes.py                LangGraph node functions + _parse_json()
    graph.py                Graph wiring + MemorySaver checkpointer
  prompts/
    extract_claim.py        Claim normalisation prompt
    analyze_images.py       Vision analysis prompt
    synthesize_decision.py  Decision synthesis prompt
  utils/
    llm.py                  LiteLLM client — 4-model fallback, TPM retry
    schema.py               Allowed values + validate_row()
    logger.py               AGENTS.md log file + onboarding gate
    csv_reader.py           Load claims, history, requirements
    image_loader.py         Encode images as base64 (auto-resize to ≤1568px)
    output_writer.py        Write output.csv with real-time flush
  strategies/
    single_shot.py          Strategy A baseline (1 LLM call per row)
  evaluation/
    main.py                 Eval harness — scores against sample_claims.csv
    evaluation_report.md    Generated operational analysis (run make eval)
    metrics.json            Generated accuracy metrics
```

## Key implementation notes

**LLM routing (`utils/llm.py`):** Uses LiteLLM with a 4-model fallback chain. Vision calls only go to vision-capable models (Gemini, llama-4-scout); text-only models (llama-3.3-70b, llama-3.1-8b) receive plain string content. TPM exhaustion triggers a wait-and-retry loop (3 attempts).

**JSON parsing (`graph/nodes.py`):** `_parse_json()` uses 3 strategies — direct parse, first `{...}` block, last `{...}` block — to handle model responses that wrap JSON in prose or markdown fences.

**Real-time output:** `output_writer.py` flushes after every row. `logger.py` flushes after every log entry. Results appear in `results/run_TIMESTAMP_<mode>/` per run and are also written to root `output.csv`.

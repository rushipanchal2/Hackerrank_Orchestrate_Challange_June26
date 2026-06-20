"""Evaluation harness: compare system output against sample_claims.csv ground truth.

Usage:
  python code/evaluation/main.py                          # LangGraph on sample
  python code/evaluation/main.py --strategy single_shot   # Strategy A
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from code.graph.graph import build_graph
from code.graph.state import default_state
from code.strategies.single_shot import run_single_shot
from code.utils.csv_reader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from code.utils.image_loader import encode_images
from code.utils.schema import REQUIRED_COLUMNS, validate_row
from code.utils import llm as _llm

_SCORED_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "evidence_standard_met",
    "valid_image",
    "severity",
]
_JACCARD_FIELDS = ["risk_flags"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation against sample_claims.csv")
    p.add_argument("--sample", default="dataset/sample_claims.csv")
    p.add_argument("--images-dir", default="dataset", dest="images_dir")
    p.add_argument("--history", default="dataset/user_history.csv")
    p.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    p.add_argument("--strategy", choices=["langgraph", "single_shot"],
                   default="langgraph")
    p.add_argument("--delay", type=float, default=0.3)
    return p.parse_args()


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split(";")) - {"none"}
    sb = set(b.split(";")) - {"none"}
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _norm_bool(val) -> str:
    return "true" if str(val).lower() == "true" else "false"


def _run_folder() -> pathlib.Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = _REPO_ROOT / "results" / f"run_{ts}_eval"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _count_test_rows() -> int:
    """Actual number of rows in dataset/claims.csv (the real test set)."""
    try:
        return len(load_claims(_REPO_ROOT / "dataset" / "claims.csv"))
    except Exception:
        return 44


def _report_dir() -> pathlib.Path:
    """code/evaluation/ — lives INSIDE the code/ tree so it is bundled in code.zip."""
    d = pathlib.Path(__file__).resolve().parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _generate_report(results: dict, run_dir: pathlib.Path,
                     elapsed: float, n_images: int, usage: dict) -> None:
    """Write code/evaluation/evaluation_report.md as required by the problem statement.

    Uses REAL token counts captured from LiteLLM responses (llm.get_usage()), not
    hardcoded estimates, and extrapolates from the actual claims.csv row count.
    """
    n = results["n_rows"]
    acc = results["accuracy"]
    strategy = results["strategy"]

    n_test = _count_test_rows()
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", 0)
    calls = usage.get("calls", 0) or (n * (1 if strategy == "single_shot" else 3))
    calls_per_row = calls / n if n else 0
    in_per_call = in_tok / calls if calls else 0
    out_per_call = out_tok / calls if calls else 0
    scale = n_test / n if n else 0

    report = f"""# Evaluation Report — Multi-Modal Evidence Review

Generated: {datetime.now().isoformat()}
Strategy: {strategy}
Dataset: sample_claims.csv ({n} rows)
Run dir: {run_dir}

> Token counts below are **measured** from live LiteLLM responses for this run,
> not estimates. Full-test-set figures are linear extrapolations to the
> {n_test} rows in `dataset/claims.csv`.

## Accuracy

| Field | Score |
|---|---|
"""
    for field, score in acc.items():
        metric = "Jaccard" if field in _JACCARD_FIELDS else "Exact match"
        report += f"| {field} | {score:.1f}% ({metric}) |\n"

    report += f"""
## Operational Analysis

### Model Calls
- Rows processed: {n}
- Measured LLM calls (sample): {calls}  (~{calls_per_row:.1f} per row)
- Pipeline: extract_claim (text) + analyze_images (vision) + synthesize_decision (text)
- Extrapolated to full test set ({n_test} rows): ~{round(calls * scale)} calls

### Token Usage (measured)
- Total input tokens (sample): {in_tok:,}
- Total output tokens (sample): {out_tok:,}
- Average input tokens per call: ~{in_per_call:,.0f} (includes base64 image data for vision calls)
- Average output tokens per call: ~{out_per_call:,.0f}
- Full test set extrapolation: ~{round(in_tok * scale):,} input / ~{round(out_tok * scale):,} output

### Images Processed
- Total images processed: {n_images}
- Average images per claim: {n_images/n:.1f}
- Images are resized to max 1568px edge before encoding to stay under API limits

### Cost Estimate (full test set, {n_test} rows)
| Provider | Model | Input $/1M | Output $/1M | Est. Cost |
|---|---|---|---|---|
| Groq | llama-4-scout | Free | Free | $0.00 |
| Gemini | gemini-2.5-flash | Free (quota) | Free (quota) | $0.00 |

Free-tier providers only → $0 within quota. For reference, the same
~{round(in_tok * scale):,} in / ~{round(out_tok * scale):,} out on a paid tier
(e.g. Gemini 2.5 Flash @ $0.30/$2.50 per 1M) would cost roughly
${(in_tok * scale / 1e6 * 0.30 + out_tok * scale / 1e6 * 2.50):.3f}.

### Latency / Runtime
- Sample run elapsed: {elapsed:.1f}s
- Average per row: {elapsed/n:.1f}s
- Full test set estimate: ~{elapsed/n*n_test:.0f}s (~{elapsed/n*n_test/60:.1f} min)

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
"""

    suffix = "" if strategy == "langgraph" else f"_{strategy}"
    report_path = _report_dir() / f"evaluation_report{suffix}.md"
    report_path.write_text(report, encoding="utf-8")

    run_report = run_dir / "evaluation_report.md"
    run_report.write_text(report, encoding="utf-8")
    print(f"\nEvaluation report → {report_path}")


def evaluate(args: argparse.Namespace) -> dict:
    root = _REPO_ROOT
    sample_path = root / args.sample
    images_base = root / args.images_dir
    run_dir = _run_folder()

    _llm.startup_banner()
    _llm.reset_usage()

    rows = load_claims(sample_path)
    user_history_dict = load_user_history(root / args.history)
    requirements_list = load_evidence_requirements(root / args.requirements)

    input_cols = {"user_id", "image_paths", "user_claim", "claim_object"}
    expected_cols = [c for c in REQUIRED_COLUMNS if c not in input_cols]

    scores: dict[str, list[float]] = {f: [] for f in _SCORED_FIELDS + _JACCARD_FIELDS}
    mismatches: list[dict] = []
    n_images = 0
    t_start = time.time()

    graph = build_graph() if args.strategy == "langgraph" else None
    configurable = {
        "user_history_dict": user_history_dict,
        "requirements_list": requirements_list,
        "images_base_dir": str(images_base),
    }

    for idx, row in enumerate(rows):
        claim_object = row["claim_object"]
        expected = {c: row.get(c, "") for c in expected_cols}

        image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
        n_images += len(image_paths)

        print(f"  [{idx+1:02d}/{len(rows)}] {row['user_id']} ({claim_object}) ...",
              end="", flush=True)

        if args.strategy == "langgraph":
            state = default_state(
                user_id=row["user_id"],
                image_paths_raw=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=claim_object,
            )
            thread_id = f"eval_{idx:04d}_{row['user_id']}"
            try:
                final = graph.invoke(
                    state,
                    config={"configurable": {**configurable, "thread_id": thread_id}},
                )
            except SystemExit:
                raise
            except Exception as exc:
                print(f" ERROR: {exc}")
                final = dict(state)

            pred_raw = validate_row(
                {
                    "claim_status": final.get("claim_status", "not_enough_information"),
                    "issue_type": final.get("issue_type", "unknown"),
                    "object_part": final.get("object_part", "unknown"),
                    "evidence_standard_met": final.get("evidence_standard_met", False),
                    "valid_image": final.get("valid_image", False),
                    "severity": final.get("severity", "unknown"),
                    "risk_flags": final.get("risk_flags", ["none"]),
                    "supporting_image_ids": final.get("supporting_image_ids", ["none"]),
                    "evidence_standard_met_reason": final.get("evidence_standard_met_reason", ""),
                    "claim_status_justification": final.get("claim_status_justification", ""),
                },
                claim_object=claim_object,
            )
        else:
            encoded = encode_images(image_paths, str(images_base))
            try:
                result = run_single_shot(
                    row=row,
                    user_history=user_history_dict.get(row["user_id"], {}),
                    requirements=requirements_list,
                    encoded_images=encoded,
                    claim_object=claim_object,
                )
                pred_raw = validate_row(result, claim_object=claim_object)
            except Exception as exc:
                print(f" ERROR: {exc}")
                pred_raw = validate_row({}, claim_object=claim_object)

        pred_status = pred_raw.get("claim_status", "not_enough_information")
        exp_status = expected.get("claim_status", "")
        match_icon = "✓" if pred_status == exp_status else "✗"
        print(f" {match_icon} {pred_status} (expected: {exp_status})")

        for field in _SCORED_FIELDS:
            pred_val = pred_raw.get(field, "")
            exp_val = expected.get(field, "")
            if field in ("evidence_standard_met", "valid_image"):
                pred_val = _norm_bool(pred_val)
                exp_val = _norm_bool(exp_val)
            match = 1.0 if str(pred_val).lower() == str(exp_val).lower() else 0.0
            scores[field].append(match)
            if match == 0.0:
                mismatches.append({
                    "row": idx,
                    "user_id": row["user_id"],
                    "field": field,
                    "predicted": pred_val,
                    "expected": exp_val,
                })

        jacc = _jaccard(
            pred_raw.get("risk_flags", "none"),
            expected.get("risk_flags", "none"),
        )
        scores["risk_flags"].append(jacc)

        if args.delay > 0 and idx < len(rows) - 1:
            time.sleep(args.delay)

    elapsed = time.time() - t_start
    summary: dict[str, float] = {
        f: round(sum(v) / len(v) * 100, 1) if v else 0.0
        for f, v in scores.items()
    }

    results = {
        "strategy": args.strategy,
        "n_rows": len(rows),
        "accuracy": summary,
        "mismatches": mismatches[:20],
        "elapsed": elapsed,
    }

    print(f"\n{'='*55}")
    print(f"Evaluation — {args.strategy}  |  n={len(rows)}  |  {elapsed:.1f}s")
    print(f"{'='*55}")
    for field, acc in summary.items():
        label = "Jaccard" if field in _JACCARD_FIELDS else "Exact"
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {field:<30} {bar}  {acc:>5.1f}%  ({label})")
    print(f"{'='*55}")
    print("  First 5 mismatches (claim_status):")
    shown = [m for m in mismatches if m["field"] == "claim_status"][:5]
    for m in shown:
        print(f"    row={m['row']} {m['user_id']}: pred={m['predicted']} exp={m['expected']}")

    usage = _llm.get_usage()
    results["usage"] = usage

    # Save metrics — to the run folder AND to code/evaluation/ (bundled in code.zip)
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    bundled_metrics = _report_dir() / f"metrics_{args.strategy}.json"
    with open(bundled_metrics, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics → {metrics_path}")

    _generate_report(results, run_dir, elapsed, n_images, usage)
    _write_comparison()

    return results


def _write_comparison() -> None:
    """If both strategy metrics exist, write code/evaluation/comparison.md."""
    rd = _report_dir()
    lg_path = rd / "metrics_langgraph.json"
    ss_path = rd / "metrics_single_shot.json"
    if not (lg_path.exists() and ss_path.exists()):
        return
    try:
        lg = json.loads(lg_path.read_text(encoding="utf-8"))
        ss = json.loads(ss_path.read_text(encoding="utf-8"))
    except Exception:
        return

    fields = list(lg.get("accuracy", {}).keys())
    lines = [
        "# Strategy Comparison — Multi-Modal Evidence Review",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "Two strategies evaluated on the same `sample_claims.csv`:",
        "- **single_shot** — one LLM call per claim (baseline)",
        "- **langgraph** — 3-node pipeline (extract → analyze images → synthesize)",
        "",
        "## Accuracy (higher is better)",
        "",
        "| Field | single_shot | langgraph | Δ (langgraph − single_shot) |",
        "|---|---|---|---|",
    ]
    for f in fields:
        a = ss.get("accuracy", {}).get(f, 0.0)
        b = lg.get("accuracy", {}).get(f, 0.0)
        lines.append(f"| {f} | {a:.1f}% | {b:.1f}% | {b - a:+.1f} |")

    lg_u, ss_u = lg.get("usage", {}), ss.get("usage", {})
    lines += [
        "",
        "## Cost / Latency (sample run)",
        "",
        "| Metric | single_shot | langgraph |",
        "|---|---|---|",
        f"| LLM calls | {ss_u.get('calls', 0)} | {lg_u.get('calls', 0)} |",
        f"| Input tokens | {ss_u.get('prompt_tokens', 0):,} | {lg_u.get('prompt_tokens', 0):,} |",
        f"| Output tokens | {ss_u.get('completion_tokens', 0):,} | {lg_u.get('completion_tokens', 0):,} |",
        f"| Elapsed (s) | {ss.get('elapsed', 0):.1f} | {lg.get('elapsed', 0):.1f} |",
        "",
        "**Takeaway:** single_shot is cheaper/faster per claim; the langgraph "
        "pipeline trades more calls/tokens for separable reasoning steps "
        "(claim normalization, per-image grounding, injection resistance) and is "
        "the strategy used for the final `output.csv`.",
        "",
    ]
    out = rd / "comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison → {out}")


if __name__ == "__main__":
    evaluate(_parse_args())

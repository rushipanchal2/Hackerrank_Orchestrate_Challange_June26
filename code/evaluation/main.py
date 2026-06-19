"""Evaluation harness: compare system output against sample_claims.csv ground truth."""

import argparse
import json
import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

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

# Fields to score (exact match)
_SCORED_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "evidence_standard_met",
    "valid_image",
    "severity",
]

# risk_flags uses Jaccard similarity (order-independent set comparison)
_JACCARD_FIELDS = ["risk_flags"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation against sample_claims.csv")
    p.add_argument("--sample", default="dataset/sample_claims.csv")
    p.add_argument("--images-dir", default="dataset", dest="images_dir")
    p.add_argument("--history", default="dataset/user_history.csv")
    p.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    p.add_argument("--strategy", choices=["langgraph", "single_shot"], default="langgraph")
    p.add_argument("--output-metrics", default="code/evaluation/metrics.json",
                   dest="output_metrics")
    p.add_argument("--delay", type=float, default=1.0)
    return p.parse_args()


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split(";")) - {"none"}
    sb = set(b.split(";")) - {"none"}
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def _normalise_bool(val) -> str:
    return "true" if str(val).lower() == "true" else "false"


def evaluate(args: argparse.Namespace) -> dict:
    root = _REPO_ROOT
    sample_path = root / args.sample
    images_base = root / args.images_dir

    rows = load_claims(sample_path)
    user_history_dict = load_user_history(root / args.history)
    requirements_list = load_evidence_requirements(root / args.requirements)

    # Split into input columns and expected output columns
    input_cols = {"user_id", "image_paths", "user_claim", "claim_object"}
    expected_cols = [c for c in REQUIRED_COLUMNS if c not in input_cols]

    scores: dict[str, list[float]] = {f: [] for f in _SCORED_FIELDS + _JACCARD_FIELDS}
    mismatches: list[dict] = []

    graph = build_graph() if args.strategy == "langgraph" else None
    configurable = {
        "user_history_dict": user_history_dict,
        "requirements_list": requirements_list,
        "images_base_dir": str(images_base),
    }

    for idx, row in enumerate(rows):
        claim_object = row["claim_object"]
        expected = {c: row.get(c, "") for c in expected_cols}

        # ── Run the selected strategy ──────────────────────────────────────
        if args.strategy == "langgraph":
            state = default_state(
                user_id=row["user_id"],
                image_paths_raw=row["image_paths"],
                user_claim=row["user_claim"],
                claim_object=claim_object,
            )
            thread_id = f"eval_{idx:04d}_{row['user_id']}"
            final = graph.invoke(
                state,
                config={"configurable": {**configurable, "thread_id": thread_id}},
            )
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
            # Strategy A: single-shot
            image_paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
            encoded = encode_images(image_paths, str(images_base))
            result = run_single_shot(
                row=row,
                user_history=user_history_dict.get(row["user_id"], {}),
                requirements=requirements_list,
                encoded_images=encoded,
                claim_object=claim_object,
            )
            pred_raw = validate_row(result, claim_object=claim_object)

        # ── Score exact-match fields ───────────────────────────────────────
        for field in _SCORED_FIELDS:
            pred_val = pred_raw.get(field, "")
            exp_val = expected.get(field, "")
            # Normalise booleans
            if field in ("evidence_standard_met", "valid_image"):
                pred_val = _normalise_bool(pred_val)
                exp_val = _normalise_bool(exp_val)
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

        # ── Score risk_flags with Jaccard ──────────────────────────────────
        jacc = _jaccard(pred_raw.get("risk_flags", "none"), expected.get("risk_flags", "none"))
        scores["risk_flags"].append(jacc)

        if args.delay > 0 and idx < len(rows) - 1:
            time.sleep(args.delay)

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary: dict[str, float] = {}
    for field, vals in scores.items():
        summary[field] = round(sum(vals) / len(vals) * 100, 1) if vals else 0.0

    results = {
        "strategy": args.strategy,
        "n_rows": len(rows),
        "accuracy": summary,
        "mismatches": mismatches[:20],  # cap to keep JSON manageable
    }

    # ── Print report ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Evaluation — strategy: {args.strategy}  |  n={len(rows)}")
    print(f"{'='*50}")
    for field, acc in summary.items():
        label = "Jaccard" if field in _JACCARD_FIELDS else "Exact match"
        print(f"  {field:<35} {acc:>5.1f}%  ({label})")
    print(f"{'='*50}")
    print(f"  First 5 mismatches:")
    for m in mismatches[:5]:
        print(f"    row={m['row']} {m['user_id']} | {m['field']}: pred={m['predicted']} exp={m['expected']}")

    # ── Save metrics ───────────────────────────────────────────────────────
    out_path = root / args.output_metrics
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {out_path}")

    return results


if __name__ == "__main__":
    evaluate(_parse_args())

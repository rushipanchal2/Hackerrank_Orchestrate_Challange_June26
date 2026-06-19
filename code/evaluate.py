"""
Evaluation harness for Multi-Modal Evidence Review.

Usage:
  python code/evaluate.py                          # Strategy B on sample, full report
  python code/evaluate.py --strategy A             # Strategy A (single-shot baseline)
  python code/evaluate.py --strategy both          # A + B + comparison.md
  python code/evaluate.py --dataset test           # Final run on claims.csv (no scoring)
  python code/evaluate.py --strategy both --no-html
"""

from __future__ import annotations

import argparse
import csv
import html as _html
import json
import pathlib
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from code.utils.csv_reader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from code.utils.image_loader import encode_images
from code.utils.schema import REQUIRED_COLUMNS, validate_row

# ── Constants ────────────────────────────────────────────────────────────────

CLAIM_STATUS_CLASSES = ["supported", "contradicted", "not_enough_information"]
INPUT_COLS = {"user_id", "image_paths", "user_claim", "claim_object"}
EXACT_FIELDS = [
    "claim_status", "issue_type", "object_part",
    "evidence_standard_met", "valid_image", "severity",
]
DATASETS = {
    "sample": ("dataset/sample_claims.csv", "images/sample"),
    "test":   ("dataset/claims.csv",        "images/test"),
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation harness")
    p.add_argument("--strategy", choices=["A", "B", "both"], default="B")
    p.add_argument("--dataset",  choices=["sample", "test"],  default="sample")
    p.add_argument("--delay",    type=float, default=1.0,
                   help="Seconds between API calls (default 1.0)")
    p.add_argument("--history",      default="dataset/user_history.csv")
    p.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    p.add_argument("--no-html",  action="store_true", dest="no_html")
    return p.parse_args()


# ── Path helpers ─────────────────────────────────────────────────────────────

def _run_dir(ts: str, strategy: str, dataset: str) -> pathlib.Path:
    p = _ROOT / "results" / f"run_{ts}_{strategy}_{dataset}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Runner ────────────────────────────────────────────────────────────────────

class Runner:
    def __init__(
        self,
        strategy: str,
        user_history_dict: dict,
        requirements_list: list,
        images_base: pathlib.Path,
        delay: float,
    ) -> None:
        self.strategy = strategy
        self.user_history_dict = user_history_dict
        self.requirements_list = requirements_list
        self.images_base = str(images_base)
        self.delay = delay
        self._graph = None

    def _graph_instance(self):
        if self._graph is None:
            from langgraph.checkpoint.memory import MemorySaver
            from code.graph.graph import build_graph
            self._graph = build_graph(checkpointer=MemorySaver())
        return self._graph

    def _run_B(self, row: dict, idx: int) -> dict:
        from code.graph.state import default_state
        state = default_state(
            user_id=row["user_id"],
            image_paths_raw=row["image_paths"],
            user_claim=row["user_claim"],
            claim_object=row["claim_object"],
        )
        cfg = {
            "configurable": {
                "user_history_dict": self.user_history_dict,
                "requirements_list": self.requirements_list,
                "images_base_dir": self.images_base,
                "thread_id": f"eval_{idx:04d}_{row['user_id']}",
            }
        }
        try:
            return self._graph_instance().invoke(state, config=cfg)
        except Exception as exc:
            return {**state, "errors": [str(exc)],
                    "claim_status": "not_enough_information", "severity": "unknown"}

    def _run_A(self, row: dict) -> dict:
        from code.strategies.single_shot import run_single_shot
        paths = [p.strip() for p in row["image_paths"].split(";") if p.strip()]
        encoded = encode_images(paths, self.images_base)
        result = run_single_shot(
            row=row,
            user_history=self.user_history_dict.get(row["user_id"], {}),
            requirements=self.requirements_list,
            encoded_images=encoded,
            claim_object=row["claim_object"],
        )
        validated = validate_row(result, claim_object=row["claim_object"])
        return {**row, "image_paths_raw": row["image_paths"], **validated}

    def run_all(self, rows: list[dict]) -> list[dict]:
        results: list[dict] = []
        n = len(rows)
        for idx, row in enumerate(rows):
            label = f"[{idx+1}/{n}] {row['user_id']} ({row['claim_object']})"
            print(f"  {label}", end="", flush=True)
            try:
                pred = self._run_B(row, idx) if self.strategy == "B" else self._run_A(row)
                results.append(pred)
                print(" ✓")
            except Exception as exc:
                print(f" ✗  {exc}")
                results.append({**row, "image_paths_raw": row["image_paths"],
                                 "claim_status": "not_enough_information", "severity": "unknown"})
            if self.delay > 0 and idx < n - 1:
                time.sleep(self.delay)
        return results


# ── Output CSV ────────────────────────────────────────────────────────────────

def save_output_csv(predictions: list[dict], path: pathlib.Path) -> None:
    from code.utils.output_writer import close_writer, open_writer, write_row
    writer, fh = open_writer(path)
    for pred in predictions:
        write_row(writer, pred)
    close_writer(writer, fh)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _norm_bool(v) -> str:
    return "true" if str(v).lower() == "true" else "false"


def _pval(pred: dict, field: str) -> str:
    v = pred.get(field, "")
    if field in ("evidence_standard_met", "valid_image"):
        return _norm_bool(v)
    if isinstance(v, list):
        v = ";".join(v)
    return str(v).lower().strip()


def _gval(gt: dict, field: str) -> str:
    v = gt.get(field, "")
    if field in ("evidence_standard_met", "valid_image"):
        return _norm_bool(v)
    return str(v).lower().strip()


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split(";")) - {"none", ""}
    sb = set(b.split(";")) - {"none", ""}
    if not sa and not sb:
        return 1.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 1.0


class Scorer:
    def __init__(self, predictions: list[dict], ground_truth: list[dict]) -> None:
        if len(predictions) != len(ground_truth):
            raise ValueError(f"Length mismatch: {len(predictions)} vs {len(ground_truth)}")
        self.preds = predictions
        self.gts = ground_truth
        self.n = len(predictions)

    # ── Field-level metrics ───────────────────────────────────────────────────

    def exact_accuracy(self, field: str) -> float:
        hits = sum(1 for p, g in zip(self.preds, self.gts) if _pval(p, field) == _gval(g, field))
        return hits / self.n * 100

    def jaccard_avg(self, field: str = "risk_flags") -> float:
        scores = [_jaccard(_pval(p, field), _gval(g, field)) for p, g in zip(self.preds, self.gts)]
        return sum(scores) / len(scores) * 100

    # ── Confusion matrix + per-class F1 ──────────────────────────────────────

    def confusion_matrix(self, field: str, classes: list[str]) -> dict[str, dict[str, int]]:
        m: dict[str, dict[str, int]] = {c: {d: 0 for d in classes} for c in classes}
        for p, g in zip(self.preds, self.gts):
            actual, predicted = _gval(g, field), _pval(p, field)
            if actual in m and predicted in m:
                m[actual][predicted] += 1
        return m

    def per_class_metrics(self, confusion: dict, classes: list[str]) -> dict:
        out = {}
        for label in classes:
            tp = confusion.get(label, {}).get(label, 0)
            fp = sum(confusion.get(c, {}).get(label, 0) for c in classes if c != label)
            fn = sum(confusion.get(label, {}).get(c, 0) for c in classes if c != label)
            support = tp + fn
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            out[label] = {"precision": round(prec, 3), "recall": round(rec, 3),
                          "f1": round(f1, 3), "support": support}
        return out

    # ── Error cases ───────────────────────────────────────────────────────────

    def error_cases(self, field: str = "claim_status") -> list[dict]:
        errors = []
        for idx, (p, g) in enumerate(zip(self.preds, self.gts)):
            pv, gv = _pval(p, field), _gval(g, field)
            if pv == gv:
                continue
            # Severity 3 = direct supported↔contradicted flip (worst)
            pair = {pv, gv}
            sev = 3 if pair == {"supported", "contradicted"} else \
                  2 if "not_enough_information" not in pair else 1
            errors.append({
                "row_index": idx,
                "user_id": g.get("user_id", ""),
                "claim_object": g.get("claim_object", ""),
                "image_paths": g.get("image_paths", ""),
                "expected": gv,
                "predicted": pv,
                "justification": str(p.get("claim_status_justification", ""))[:250],
                "severity": sev,
            })
        return sorted(errors, key=lambda x: -x["severity"])

    # ── Full metrics dict ─────────────────────────────────────────────────────

    def compute(self) -> dict:
        accuracy = {
            f: round(self.exact_accuracy(f), 1) for f in EXACT_FIELDS
        }
        accuracy["risk_flags_jaccard"] = round(self.jaccard_avg("risk_flags"), 1)

        confusion = self.confusion_matrix("claim_status", CLAIM_STATUS_CLASSES)
        per_class = self.per_class_metrics(confusion, CLAIM_STATUS_CLASSES)
        macro_f1  = round(sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1), 3)

        # Per-claim_object breakdown of object_part accuracy
        by_object: dict[str, list] = defaultdict(list)
        for p, g in zip(self.preds, self.gts):
            obj = g.get("claim_object", "")
            by_object[obj].append(_pval(p, "object_part") == _gval(g, "object_part"))
        object_part_by_type = {
            obj: round(sum(hits) / len(hits) * 100, 1)
            for obj, hits in by_object.items() if hits
        }

        errors = self.error_cases()
        return {
            "n_rows": self.n,
            "accuracy": accuracy,
            "claim_status": {
                "confusion_matrix": confusion,
                "per_class": per_class,
                "macro_f1": macro_f1,
            },
            "object_part_by_type": object_part_by_type,
            "error_cases": errors[:10],
        }


# ── Comparison markdown ───────────────────────────────────────────────────────

def build_comparison_md(
    ts: str,
    metrics_A: dict,
    metrics_B: dict,
    preds_A: list[dict],
    preds_B: list[dict],
    ground_truth: list[dict],
) -> str:
    lines = [
        "# Strategy Comparison\n",
        f"Generated: {ts.replace('_', '-', 1).replace('_', ' ', 1)}\n",
        "| Field | Strategy A (single-shot) | Strategy B (LangGraph) | Δ | Winner |",
        "|-------|:------------------------:|:----------------------:|:---:|:------:|",
    ]
    acc_A = metrics_A["accuracy"]
    acc_B = metrics_B["accuracy"]
    field_labels = {
        "claim_status":       "claim_status",
        "issue_type":         "issue_type",
        "object_part":        "object_part",
        "evidence_standard_met": "evidence_standard_met",
        "valid_image":        "valid_image",
        "severity":           "severity",
        "risk_flags_jaccard": "risk_flags (Jaccard)",
    }
    overall_A, overall_B, n_fields = 0.0, 0.0, 0
    for key, label in field_labels.items():
        a, b = acc_A.get(key, 0.0), acc_B.get(key, 0.0)
        delta = b - a
        winner = "**B**" if b > a else ("**A**" if a > b else "tie")
        lines.append(f"| {label} | {a:.1f}% | {b:.1f}% | {delta:+.1f} | {winner} |")
        overall_A += a
        overall_B += b
        n_fields += 1

    oa = overall_A / n_fields
    ob = overall_B / n_fields
    od = ob - oa
    ow = "**B**" if ob > oa else ("**A**" if oa > ob else "tie")
    lines += [
        f"| **Overall average** | **{oa:.1f}%** | **{ob:.1f}%** | **{od:+.1f}** | {ow} |",
        "",
        f"macro-F1 (claim_status): A={metrics_A['claim_status']['macro_f1']:.3f}  "
        f"B={metrics_B['claim_status']['macro_f1']:.3f}\n",
    ]

    # Per-case breakdown
    b_better, a_better = [], []
    for idx, (pa, pb, g) in enumerate(zip(preds_A, preds_B, ground_truth)):
        ev = _gval(g, "claim_status")
        av = _pval(pa, "claim_status")
        bv = _pval(pb, "claim_status")
        uid = g.get("user_id", f"row_{idx}")
        if bv == ev and av != ev:
            b_better.append(f"  - row {idx} `{uid}` — A:`{av}` B:`{bv}` GT:`{ev}`")
        elif av == ev and bv != ev:
            a_better.append(f"  - row {idx} `{uid}` — A:`{av}` B:`{bv}` GT:`{ev}`")

    if b_better:
        lines += ["## B correct, A wrong (claim_status)\n"] + b_better + [""]
    if a_better:
        lines += ["## A correct, B wrong (claim_status)\n"] + a_better + [""]

    return "\n".join(lines)


# ── HTML report ───────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.6rem; color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: .4rem; }
h2 { font-size: 1.15rem; color: #2c4a6e; margin-top: 2rem; }
h3 { font-size: 1rem; color: #444; margin-top: 1.5rem; }
.meta { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: .85rem;
        color: #555; margin-bottom: 1.5rem; }
.meta span { background: #e8f0fe; padding: .2rem .7rem; border-radius: 12px; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .9rem; }
th { background: #1a3a5c; color: #fff; padding: 7px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e4e4e4; vertical-align: top; }
tr:hover td { background: #f0f4ff; }
.hi  { background: #d4edda; }
.mid { background: #fff3cd; }
.lo  { background: #f8d7da; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.conf-diag { background: #c3e6cb; font-weight: 700; text-align: center; }
.conf-off  { background: #f5c6cb; text-align: center; }
.conf-zero { background: #f5f5f5; text-align: center; color: #999; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: .8rem;
         font-weight: 600; }
.badge-supported    { background: #c3e6cb; color: #155724; }
.badge-contradicted { background: #f5c6cb; color: #721c24; }
.badge-nei          { background: #ffeaa7; color: #6c5f00; }
.monospace { font-family: monospace; font-size: .8rem; color: #444; word-break: break-all; }
.sev3 { background: #f8d7da; }
.sev2 { background: #fff3cd; }
.sev1 { background: #e2f0fb; }
footer { margin-top: 3rem; font-size: .8rem; color: #888; text-align: center; }
"""


def _acc_cls(acc: float) -> str:
    return "hi" if acc >= 80 else ("mid" if acc >= 60 else "lo")


def _badge(label: str) -> str:
    cls = {"supported": "badge-supported", "contradicted": "badge-contradicted"}.get(
        label, "badge-nei"
    )
    return f'<span class="badge {cls}">{_html.escape(label)}</span>'


def _confusion_html(confusion: dict, classes: list[str], totals: dict[str, int]) -> str:
    short = {"supported": "S", "contradicted": "C", "not_enough_information": "N"}
    rows = ["<table>",
            "<tr><th>Actual \\ Predicted</th>" +
            "".join(f"<th>{short.get(c, c)}</th>" for c in classes) + "<th>Total</th></tr>"]
    for actual in classes:
        row_total = totals.get(actual, 0)
        cells = []
        for predicted in classes:
            val = confusion.get(actual, {}).get(predicted, 0)
            if actual == predicted:
                cells.append(f'<td class="conf-diag">{val}</td>')
            elif val > 0:
                pct = val / row_total * 100 if row_total else 0
                cells.append(f'<td class="conf-off">{val}<br><small>{pct:.0f}%</small></td>')
            else:
                cells.append(f'<td class="conf-zero">0</td>')
        rows.append(f"<tr><td><b>{_badge(actual)}</b></td>{''.join(cells)}<td class='num'>{row_total}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _metrics_table_html(accuracy: dict) -> str:
    label_map = {
        "claim_status":          "claim_status (primary)",
        "issue_type":            "issue_type",
        "object_part":           "object_part",
        "evidence_standard_met": "evidence_standard_met",
        "valid_image":           "valid_image",
        "severity":              "severity",
        "risk_flags_jaccard":    "risk_flags (Jaccard)",
    }
    rows = ["<table>", "<tr><th>Field</th><th>Score</th><th>Rating</th></tr>"]
    for key, label in label_map.items():
        acc = accuracy.get(key, 0.0)
        cls = _acc_cls(acc)
        bar_w = int(acc * 1.5)
        bar = f'<div style="background:{"#28a745" if cls=="hi" else "#ffc107" if cls=="mid" else "#dc3545"};height:8px;width:{bar_w}px;border-radius:4px;display:inline-block;"></div>'
        rows.append(
            f"<tr><td>{label}</td>"
            f'<td class="num {cls}"><b>{acc:.1f}%</b></td>'
            f"<td>{bar}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _f1_table_html(per_class: dict) -> str:
    rows = ["<table>",
            "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>"]
    for label, m in per_class.items():
        f1_cls = _acc_cls(m["f1"] * 100)
        rows.append(
            f"<tr><td>{_badge(label)}</td>"
            f"<td class='num'>{m['precision']:.3f}</td>"
            f"<td class='num'>{m['recall']:.3f}</td>"
            f"<td class='num {f1_cls}'><b>{m['f1']:.3f}</b></td>"
            f"<td class='num'>{m['support']}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _errors_table_html(errors: list[dict]) -> str:
    if not errors:
        return "<p>No misclassifications. 🎉</p>"
    rows = ["<table>",
            "<tr><th>#</th><th>user_id</th><th>object</th>"
            "<th>Expected</th><th>Predicted</th><th>Justification</th><th>Images</th></tr>"]
    sev_cls = {3: "sev3", 2: "sev2", 1: "sev1"}
    for e in errors:
        sc = sev_cls.get(e["severity"], "")
        rows.append(
            f"<tr class='{sc}'>"
            f"<td class='num'>{e['row_index']}</td>"
            f"<td>{_html.escape(e['user_id'])}</td>"
            f"<td>{_html.escape(e['claim_object'])}</td>"
            f"<td>{_badge(e['expected'])}</td>"
            f"<td>{_badge(e['predicted'])}</td>"
            f"<td>{_html.escape(e['justification'])}</td>"
            f"<td class='monospace'>{_html.escape(e['image_paths'])}</td>"
            "</tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def generate_html_report(
    run_dir: pathlib.Path,
    metrics: dict,
    strategy: str,
    dataset: str,
    ts: str,
    scored: bool,
) -> pathlib.Path:
    acc = metrics.get("accuracy", {})
    n = metrics.get("n_rows", 0)

    if scored:
        confusion = metrics["claim_status"]["confusion_matrix"]
        per_class = metrics["claim_status"]["per_class"]
        macro_f1  = metrics["claim_status"]["macro_f1"]
        totals = {c: sum(confusion.get(c, {}).values()) for c in CLAIM_STATUS_CLASSES}
        obj_table_rows = "".join(
            f"<tr><td>{_html.escape(obj)}</td><td class='num {_acc_cls(v)}'>{v:.1f}%</td></tr>"
            for obj, v in metrics.get("object_part_by_type", {}).items()
        )
        metrics_section = f"""
<h2>Summary Metrics</h2>
{_metrics_table_html(acc)}

<h2>claim_status — Confusion Matrix</h2>
<p>Legend: <b>S</b> = supported &nbsp; <b>C</b> = contradicted &nbsp; <b>N</b> = not_enough_information</p>
{_confusion_html(confusion, CLAIM_STATUS_CLASSES, totals)}
<p>Macro-F1: <b>{macro_f1:.3f}</b></p>

<h2>claim_status — Per-Class F1</h2>
{_f1_table_html(per_class)}

<h2>object_part Accuracy by Claim Object</h2>
<table><tr><th>Claim object</th><th>Accuracy</th></tr>{obj_table_rows}</table>

<h2>Worst Misclassifications (claim_status)</h2>
<p>Severity: <span class='sev3'>■</span> supported↔contradicted (worst) &nbsp;
   <span class='sev2'>■</span> missed clear case &nbsp;
   <span class='sev1'>■</span> over-confident</p>
{_errors_table_html(metrics.get("error_cases", []))}
"""
    else:
        metrics_section = """
<p><em>Test dataset run — no ground-truth labels available. Metrics not computed.</em></p>
<p>Output written to <code>output.csv</code> in this run directory.</p>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Report — {_html.escape(strategy)} — {ts}</title>
  <style>{_CSS}</style>
</head>
<body>
<h1>Evidence Review Evaluation Report</h1>
<div class="meta">
  <span>Strategy: <b>{_html.escape(strategy)}</b></span>
  <span>Dataset: <b>{_html.escape(dataset)}</b></span>
  <span>Run: {ts}</span>
  <span>Rows: {n}</span>
</div>
{metrics_section}
<footer>Generated by code/evaluate.py &mdash; Multi-Modal Evidence Review</footer>
</body>
</html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


# ── Single run orchestration ──────────────────────────────────────────────────

def run_one(
    strategy: str,
    dataset: str,
    ts: str,
    user_history_dict: dict,
    requirements_list: list,
    images_base: pathlib.Path,
    claims_path: pathlib.Path,
    delay: float,
    gen_html: bool,
) -> tuple[dict | None, list[dict], list[dict], pathlib.Path]:
    """Run one strategy. Returns (metrics|None, predictions, ground_truth, run_dir)."""
    print(f"\n{'='*60}")
    print(f"  Strategy {strategy} | dataset={dataset}")
    print(f"{'='*60}")

    rows = load_claims(claims_path)
    runner = Runner(strategy, user_history_dict, requirements_list, images_base, delay)
    predictions = runner.run_all(rows)

    run_dir = _run_dir(ts, strategy, dataset)
    out_csv = run_dir / "output.csv"
    save_output_csv(predictions, out_csv)
    print(f"\n  → {out_csv}")

    has_labels = dataset == "sample"
    metrics: dict | None = None

    if has_labels:
        ground_truth = rows   # sample_claims.csv has all 14 columns
        scorer = Scorer(predictions, ground_truth)
        metrics = scorer.compute()
        metrics["strategy"] = strategy
        metrics["dataset"] = dataset
        metrics["timestamp"] = ts

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  → {metrics_path}")
        _print_summary(metrics)

        if gen_html:
            report_path = generate_html_report(run_dir, metrics, strategy, dataset, ts, scored=True)
            print(f"  → {report_path}")
    elif gen_html:
        report_path = generate_html_report(
            run_dir, {"n_rows": len(rows), "accuracy": {}}, strategy, dataset, ts, scored=False
        )
        print(f"  → {report_path}")

    return metrics, predictions, rows, run_dir


def _print_summary(metrics: dict) -> None:
    acc = metrics["accuracy"]
    print(f"\n  {'Field':<35} {'Score':>7}")
    print(f"  {'-'*43}")
    labels = [
        ("claim_status", "claim_status"),
        ("issue_type",   "issue_type"),
        ("object_part",  "object_part"),
        ("risk_flags_jaccard", "risk_flags (Jaccard)"),
        ("evidence_standard_met", "evidence_standard_met"),
        ("valid_image",  "valid_image"),
        ("severity",     "severity"),
    ]
    for key, label in labels:
        v = acc.get(key, 0.0)
        bar = "█" * int(v // 10)
        print(f"  {label:<35} {v:>5.1f}%  {bar}")
    print(f"\n  macro-F1 (claim_status): {metrics['claim_status']['macro_f1']:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    claims_csv, _ = DATASETS[args.dataset]
    claims_path = _ROOT / claims_csv
    images_base = _ROOT / "dataset"

    # Select LLM provider (LLM_PROVIDER=auto|gemini|groq|anthropic); aborts with a
    # clear message if none work.
    from code.utils import llm
    llm.select_provider()
    print(f"Loading reference data …")
    user_history_dict = load_user_history(_ROOT / args.history)
    requirements_list = load_evidence_requirements(_ROOT / args.requirements)
    print(f"  {len(user_history_dict)} users | {len(requirements_list)} requirements")

    gen_html = not args.no_html
    strategies = ["A", "B"] if args.strategy == "both" else [args.strategy]

    all_metrics: dict[str, dict] = {}
    all_preds:   dict[str, list[dict]] = {}
    all_gt:      list[dict] = []

    for strat in strategies:
        metrics, preds, gt_rows, _ = run_one(
            strategy=strat,
            dataset=args.dataset,
            ts=ts,
            user_history_dict=user_history_dict,
            requirements_list=requirements_list,
            images_base=images_base,
            claims_path=claims_path,
            delay=args.delay,
            gen_html=gen_html,
        )
        if metrics is not None:
            all_metrics[strat] = metrics
        all_preds[strat] = preds
        all_gt = gt_rows

    # Comparison (only when both ran with ground truth)
    if args.strategy == "both" and "A" in all_metrics and "B" in all_metrics:
        comp_md = build_comparison_md(
            ts=ts,
            metrics_A=all_metrics["A"],
            metrics_B=all_metrics["B"],
            preds_A=all_preds["A"],
            preds_B=all_preds["B"],
            ground_truth=all_gt,
        )
        comp_path = _ROOT / "results" / "comparison.md"
        comp_path.write_text(comp_md, encoding="utf-8")
        print(f"\n  → {comp_path}")
        print("\n" + comp_md[:800])

    print(f"\nDone. Results in {_ROOT / 'results'}/")


if __name__ == "__main__":
    main()

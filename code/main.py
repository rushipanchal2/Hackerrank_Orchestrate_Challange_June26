"""Main entry point: run the LangGraph pipeline on claims.csv → output.csv.

Usage:
  python code/main.py                   # full run on claims.csv
  python code/main.py --test 5          # quick test on first 5 rows
  python code/main.py --sample          # run on sample_claims.csv
  python code/main.py --skip-onboarding # skip interactive agreement gate
"""

import argparse
import csv
import pathlib
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from code.graph.graph import build_graph
from code.graph.state import default_state
from code.utils.csv_reader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from code.utils.logger import ensure_onboarding, log_path, write_row_log, write_turn
from code.utils.output_writer import close_writer, open_writer, write_row

console = Console()
_table_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Modal Evidence Review — inference")
    p.add_argument("--claims", default="dataset/claims.csv")
    p.add_argument("--output", default="output.csv")
    p.add_argument("--images-dir", default="dataset", dest="images_dir")
    p.add_argument("--history", default="dataset/user_history.csv")
    p.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    p.add_argument("--skip-onboarding", action="store_true", dest="skip_onboarding")
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--test", type=int, default=0, metavar="N",
                   help="Quick test: process only first N rows")
    p.add_argument("--sample", action="store_true",
                   help="Run on sample_claims.csv instead of claims.csv")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in submission/output.csv "
                        "(only process missing ones, then merge). Useful when a "
                        "prior run was interrupted by rate limits.")
    p.add_argument("--threads", type=int, default=2,
                   help="Parallel worker threads (default 2)")
    return p.parse_args()


def _clean_db_files() -> None:
    for name in ("checkpoints.db", "checkpoints.db-shm", "checkpoints.db-wal"):
        p = pathlib.Path(name)
        if p.exists():
            p.unlink()


def _cleanup_old_runs(keep: int = 3) -> None:
    """Delete old output/run_* folders, keeping the most recent `keep` runs."""
    output_dir = _REPO_ROOT / "output"
    if not output_dir.exists():
        return
    runs = sorted(output_dir.glob("run_*"), key=lambda p: p.name)
    for old in runs[:-keep] if len(runs) > keep else []:
        shutil.rmtree(old, ignore_errors=True)


def _run_folder(mode: str) -> pathlib.Path:
    """Create timestamped output/run_TIMESTAMP_MODE/ folder."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = _REPO_ROOT / "output" / f"run_{ts}_{mode}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _status_color(status: str) -> str:
    return {"supported": "green", "contradicted": "red",
            "not_enough_information": "yellow"}.get(status, "white")


def _build_table(rows_done: list[dict], total: int) -> Table:
    t = Table(show_header=True, header_style="bold", box=None,
              title=f"Claims Progress: {len(rows_done)}/{total}")
    t.add_column("#", width=4)
    t.add_column("User", width=10)
    t.add_column("Model", width=36)
    t.add_column("Status", width=22)
    t.add_column("Sev", width=6)
    t.add_column("E", width=3)

    for r in rows_done[-20:]:
        status = r.get("claim_status", "?")
        color = _status_color(status)
        errs = len(r.get("_errors", []))
        t.add_row(
            str(r["_idx"]),
            r.get("_user_id", "?")[:10],
            r.get("_model", "?")[:36],
            Text(status, style=color),
            r.get("severity", "?")[:6],
            str(errs) if errs else "",
        )
    return t


def _process_row(idx: int, row: dict, graph, configurable: dict) -> dict:
    """Process a single claim row. Runs in a thread pool worker."""
    from code.utils import llm

    thread_id = f"{idx:04d}_{row['user_id']}"
    state = default_state(
        user_id=row["user_id"],
        image_paths_raw=row["image_paths"],
        user_claim=row["user_claim"],
        claim_object=row["claim_object"],
    )

    try:
        final = graph.invoke(
            state,
            config={"configurable": {**configurable, "thread_id": thread_id}},
        )
        model_used = llm.active_model()
        errors = final.get("errors", [])
    except SystemExit:
        raise
    except Exception as exc:
        model_used = llm.active_model()
        errors = [str(exc)]
        final = dict(state)
        final["claim_status_justification"] = f"Processing error: {exc}"

    final["_idx"] = idx
    final["_model"] = model_used
    final["_errors"] = errors
    final["_user_id"] = row["user_id"]
    return final


def main() -> None:
    args = parse_args()
    ensure_onboarding(skip=args.skip_onboarding)
    _clean_db_files()
    _cleanup_old_runs(keep=3)

    # ── Determine mode and paths ──────────────────────────────────────────────
    if args.sample:
        mode = "sample"
        claims_file = "dataset/sample_claims.csv"
    elif args.test > 0:
        mode = f"test{args.test}"
        claims_file = args.claims
    else:
        mode = "full"
        claims_file = args.claims

    run_dir = _run_folder(mode)

    root = _REPO_ROOT
    submission_dir    = root / "submission"
    submission_dir.mkdir(exist_ok=True)
    claims_path       = root / claims_file
    output_path       = submission_dir / "output.csv"   # submission/output.csv
    run_output_path   = run_dir / "output.csv"          # output/run_.../output.csv
    images_base       = root / args.images_dir
    history_path      = root / args.history
    requirements_path = root / args.requirements

    # ── LiteLLM startup ──────────────────────────────────────────────────────
    from code.utils import llm
    console.print(f"\n[bold]Multi-Modal Evidence Review[/bold]  mode=[cyan]{mode}[/cyan]")
    llm.startup_banner()
    llm.reset_usage()

    # ── Load data ─────────────────────────────────────────────────────────────
    rows = load_claims(claims_path)
    if args.test > 0:
        rows = rows[:args.test]
        console.print(f"  [yellow]--test {args.test}: first {len(rows)} rows[/yellow]")

    user_history_dict = load_user_history(history_path)
    requirements_list = load_evidence_requirements(requirements_path)

    # ── Resume: load already-completed rows (keyed by image_paths) ────────────
    done_rows: dict[str, dict] = {}
    if args.resume and output_path.exists():
        with open(output_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_rows[r.get("image_paths", "")] = r
        pending = [(i, row) for i, row in enumerate(rows)
                   if row["image_paths"] not in done_rows]
        console.print(
            f"  [green]--resume:[/green] {len(done_rows)} rows already done, "
            f"{len(pending)} pending"
        )
    else:
        pending = list(enumerate(rows))

    console.print(
        f"  {len(rows)} claims | {len(user_history_dict)} users "
        f"| {len(requirements_list)} requirements | [cyan]{args.threads} threads[/cyan]\n"
    )

    graph = build_graph()
    configurable = {
        "user_history_dict": user_history_dict,
        "requirements_list": requirements_list,
        "images_base_dir": str(images_base),
    }

    # ── Run with thread pool ──────────────────────────────────────────────────
    results: dict[int, dict] = {}
    errors_total = 0
    t_start = time.time()
    rows_done_display: list[dict] = []
    aborted = False

    with Live(console=console, refresh_per_second=4, transient=False) as live:
        try:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                futures = {
                    pool.submit(_process_row, idx, row, graph, configurable): idx
                    for idx, row in pending
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                    except SystemExit as exc:
                        live.stop()
                        console.print(f"\n[bold red]{exc}[/bold red]")
                        aborted = True
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as exc:
                        result = {
                            "_idx": idx,
                            "_model": llm.active_model(),
                            "_errors": [str(exc)],
                            "_user_id": rows[idx]["user_id"],
                            "claim_status": "not_enough_information",
                            "severity": "unknown",
                        }

                    results[idx] = result
                    errors_total += len(result.get("_errors", []))

                    with _table_lock:
                        rows_done_display.append(result)
                        rows_done_display.sort(key=lambda r: r["_idx"])
                        live.update(_build_table(rows_done_display, len(rows)))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            aborted = True

    if aborted and not results and not done_rows:
        sys.exit(1)

    # ── Write CSVs in original claims order, merging resumed rows ─────────────
    writer,      fh      = open_writer(run_output_path)
    writer_root, fh_root = open_writer(output_path)

    for idx, row in enumerate(rows):
        if idx in results:
            r = results[idx]
            write_row(writer, r)
            write_row(writer_root, r)
            write_row_log(idx, r.get("_user_id", "?"), r.get("_model", "?"),
                          r.get("claim_status", "?"), r.get("_errors", []))
        elif row["image_paths"] in done_rows:
            raw = done_rows[row["image_paths"]]
            writer.writerow(raw);      writer._fh.flush()
            writer_root.writerow(raw); writer_root._fh.flush()

    close_writer(writer, fh)
    close_writer(writer_root, fh_root)

    # ── Copy log.txt into run folder and submission folder ───────────────────
    src_log = pathlib.Path(log_path())
    if src_log.exists():
        shutil.copy2(src_log, run_dir / "log.txt")
        shutil.copy2(src_log, submission_dir / "log.txt")

    elapsed = time.time() - t_start

    # ── Run summary ───────────────────────────────────────────────────────────
    cs_counts = Counter(r.get("claim_status", "?") for r in results.values())
    usage = llm.get_usage()
    summary_lines = [
        f"Mode: {mode}",
        f"Rows: {len(results)}/{len(rows)}",
        f"Threads: {args.threads}",
        f"Elapsed: {elapsed:.1f}s  ({len(results)/max(elapsed, 1):.2f} rows/s)",
        f"Model: {llm.active_model()}",
        f"claim_status: {dict(cs_counts)}",
        f"LLM calls: {usage.get('calls', 0)}  "
        f"(in {usage.get('prompt_tokens', 0):,} tok / out {usage.get('completion_tokens', 0):,} tok)",
        f"Node-level errors: {errors_total}",
        f"Run dir: {run_dir}",
        f"Output: {run_output_path}",
    ]
    summary = "\n".join(summary_lines)
    (run_dir / "run_summary.txt").write_text(summary + "\n", encoding="utf-8")

    console.print(f"\n[bold green]Done![/bold green]")
    for line in summary_lines:
        console.print(f"  {line}")

    if errors_total:
        console.print(
            f"\n[yellow]WARNING: {errors_total} node-level errors — "
            "inspect run_summary.txt before submitting.[/yellow]"
        )

    console.print(f"\n  Output (run)        → [cyan]{run_output_path}[/cyan]")
    console.print(f"  Output (submission) → [cyan]{output_path}[/cyan]")
    console.print(f"  Log (submission)    → [cyan]{submission_dir / 'log.txt'}[/cyan]")
    console.print(f"  Log (global)        → [cyan]{src_log}[/cyan]\n")

    write_turn(
        title=f"Run inference ({mode})",
        user_prompt=f"python code/main.py --test {args.test}" if args.test else "python code/main.py",
        summary=summary,
        actions=[
            f"Read {claims_path}",
            f"Wrote {run_output_path}",
            f"Wrote {output_path}",
            f"Copied log → {run_dir / 'log.txt'}",
        ],
    )


if __name__ == "__main__":
    main()

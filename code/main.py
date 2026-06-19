"""Main entry point: run the LangGraph pipeline on claims.csv → output.csv.

Multi-threaded execution: available providers (gemini, groq, anthropic) each
get a dedicated worker thread that processes claims in round-robin order.
This gives ~2-3x speed-up when multiple API keys are configured.

Real-time logging: every processed row is written to the AGENTS.md log file
immediately after processing, and the log is flushed to disk on each write.
"""

import argparse
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tqdm import tqdm

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from code.graph.graph import build_graph
from code.graph.state import default_state
from code.utils.csv_reader import (
    load_claims,
    load_evidence_requirements,
    load_user_history,
)
from code.utils.logger import ensure_onboarding, write_row_log, write_turn
from code.utils.output_writer import close_writer, open_writer, write_row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-Modal Evidence Review — inference")
    p.add_argument("--claims", default="dataset/claims.csv")
    p.add_argument("--output", default="output.csv")
    p.add_argument("--images-dir", default="dataset", dest="images_dir")
    p.add_argument("--history", default="dataset/user_history.csv")
    p.add_argument("--requirements", default="dataset/evidence_requirements.csv")
    p.add_argument("--fresh", action="store_true",
                   help="Delete checkpoints.db before run (legacy; always fresh now)")
    p.add_argument("--skip-onboarding", action="store_true", dest="skip_onboarding")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds to sleep between rows per worker thread (default 0.5)")
    p.add_argument("--workers", type=int, default=0,
                   help="Number of parallel workers (0 = one per available provider)")
    return p.parse_args()


def _clean_db_files() -> None:
    """Remove SQLite checkpoint files (db, WAL shm/wal) — not needed between runs."""
    for name in ("checkpoints.db", "checkpoints.db-shm", "checkpoints.db-wal"):
        p = pathlib.Path(name)
        if p.exists():
            p.unlink()
            print(f"  Removed stale file: {name}")


def _process_row(idx: int, row: dict, provider: str,
                 configurable: dict, graph,
                 delay: float, progress_lock: threading.Lock,
                 pbar) -> dict:
    """Process a single claim row. Returns the final state dict."""
    thread_id = f"{idx:04d}_{row['user_id']}_{provider}"
    state = default_state(
        user_id=row["user_id"],
        image_paths_raw=row["image_paths"],
        user_claim=row["user_claim"],
        claim_object=row["claim_object"],
    )

    from code.utils import llm as _llm
    model = _llm._CFG[provider]["model"]

    try:
        final = graph.invoke(
            state,
            config={"configurable": {**configurable, "thread_id": thread_id,
                                     "provider": provider}},
        )
        errors = final.get("errors", [])
        status = "ok" if not errors else f"errors({len(errors)})"
        write_row_log(idx, row["user_id"], provider, model, status, errors)

        with progress_lock:
            pbar.update(1)
            pbar.set_postfix({"provider": provider, "row": idx})

        if delay > 0:
            time.sleep(delay)
        return {"idx": idx, "final": final, "error": None}

    except Exception as exc:
        write_row_log(idx, row["user_id"], provider, model,
                      f"EXCEPTION: {type(exc).__name__}", [str(exc)])
        fallback = dict(state)
        fallback["claim_status_justification"] = f"Processing error: {exc}"

        with progress_lock:
            pbar.update(1)

        return {"idx": idx, "final": fallback, "error": str(exc)}


def main() -> None:
    args = parse_args()

    ensure_onboarding(skip=args.skip_onboarding)

    print("\n" + "=" * 64)
    print("  Multi-Modal Evidence Review — inference")
    print("=" * 64)

    # Always start clean — remove stale checkpoint and previous output
    _clean_db_files()
    prev_output = _REPO_ROOT / args.output
    if prev_output.exists():
        prev_output.unlink()
        print(f"  Removed previous output: {prev_output.name}")

    # Provider selection — discovers all working providers for multi-threading
    from code.utils import llm
    print("\nSelecting LLM providers …")
    llm.select_provider()
    available = llm.get_available_providers()

    n_workers = args.workers if args.workers > 0 else max(1, len(available))
    print(f"\n  Workers: {n_workers}  |  Providers: {available}")
    print(f"  Round-robin assignment: row % {len(available)} → provider")
    print(f"  Inter-row delay per worker: {args.delay}s")

    # Load data
    root = _REPO_ROOT
    claims_path       = root / args.claims
    output_path       = root / args.output
    images_base       = root / args.images_dir
    history_path      = root / args.history
    requirements_path = root / args.requirements

    print(f"\nLoading data from {claims_path.name} …")
    rows = load_claims(claims_path)
    user_history_dict = load_user_history(history_path)
    requirements_list = load_evidence_requirements(requirements_path)
    print(f"  {len(rows)} claims | {len(user_history_dict)} users "
          f"| {len(requirements_list)} requirements")

    # Each worker gets its own graph instance (MemorySaver, thread-safe)
    graphs = {p: build_graph() for p in available}

    configurable = {
        "user_history_dict": user_history_dict,
        "requirements_list": requirements_list,
        "images_base_dir": str(images_base),
    }

    # Build work items: assign provider round-robin by index
    work = [
        (idx, row, available[idx % len(available)])
        for idx, row in enumerate(rows)
    ]

    # Announce assignment
    print("\n  Provider assignment:")
    for p in available:
        assigned = [idx for idx, _, ap in work if ap == p]
        print(f"    {p} ({llm._CFG[p]['model']}): "
              f"{len(assigned)} rows → indices {assigned[:5]}{'…' if len(assigned) > 5 else ''}")

    writer, fh = open_writer(output_path)
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    results: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    errors_total = 0

    print(f"\nProcessing {len(rows)} claims with {n_workers} workers …\n")
    t_start = time.time()

    with tqdm(total=len(rows), desc="Claims", unit="row") as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(
                    _process_row,
                    idx, row, provider,
                    configurable, graphs[provider],
                    args.delay, progress_lock, pbar,
                ): idx
                for idx, row, provider in work
            }

            for future in as_completed(future_map):
                result = future.result()
                results[result["idx"]] = result

    # Write output in original row order
    for r in results:
        with write_lock:
            write_row(writer, r["final"])
        if r["error"]:
            errors_total += 1
            print(f"\n  ERROR row {r['idx']} ({rows[r['idx']]['user_id']}): {r['error']}")

    close_writer(writer, fh)

    elapsed = time.time() - t_start
    summary = (
        f"Processed {len(rows)} claims in {elapsed:.1f}s "
        f"({len(rows)/elapsed:.2f} rows/s) → {output_path}\n"
        f"Providers used: {available}\n"
        f"Workers: {n_workers}\n"
        f"Node-level errors: {errors_total}"
    )
    print(f"\n{summary}")

    if errors_total:
        print(f"\nWARNING: {errors_total} rows had errors — inspect output.csv before submitting.")

    from code.utils.logger import log_path
    print("\n" + "=" * 64)
    print("  RESULTS SAVED")
    print(f"  Output (submit this): {output_path}")
    print(f"  Log file:             {log_path()}")
    print("=" * 64 + "\n")

    write_turn(
        title="Run inference on claims.csv (multi-threaded)",
        user_prompt="python code/main.py",
        summary=summary,
        actions=[
            f"Read {claims_path}",
            f"Wrote {output_path}",
            f"Providers: {available}",
            f"Workers: {n_workers}",
            f"Log: {log_path()}",
        ],
    )


if __name__ == "__main__":
    main()

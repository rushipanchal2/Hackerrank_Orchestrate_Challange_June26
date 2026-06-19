"""CSV loading utilities. All data is loaded once and passed via LangGraph config."""

import csv
import pathlib
from typing import Any


def _open(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")
    return p


def load_claims(path: str | pathlib.Path) -> list[dict[str, str]]:
    """Return all rows from claims.csv or sample_claims.csv."""
    with open(_open(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_user_history(path: str | pathlib.Path) -> dict[str, dict[str, Any]]:
    """Return a dict keyed by user_id."""
    result: dict[str, dict[str, Any]] = {}
    with open(_open(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["user_id"]] = dict(row)
    return result


def load_evidence_requirements(path: str | pathlib.Path) -> list[dict[str, str]]:
    """Return all rows from evidence_requirements.csv."""
    with open(_open(path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

"""Output CSV writer that enforces schema column order."""

import csv
import pathlib

from code.utils.schema import REQUIRED_COLUMNS, validate_row


def open_writer(path: str | pathlib.Path):
    """Open output.csv and write the header. Returns (writer, file_handle)."""
    fh = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    return writer, fh


def write_row(writer: csv.DictWriter, state: dict) -> None:
    """Validate and write one row. Uses image_paths_raw for verbatim path echo."""
    row = {
        "user_id": state.get("user_id", ""),
        "image_paths": state.get("image_paths_raw", ""),   # verbatim original string
        "user_claim": state.get("user_claim", ""),
        "claim_object": state.get("claim_object", ""),
        "evidence_standard_met": state.get("evidence_standard_met", False),
        "evidence_standard_met_reason": state.get("evidence_standard_met_reason", ""),
        "risk_flags": state.get("risk_flags", ["none"]),
        "issue_type": state.get("issue_type", "unknown"),
        "object_part": state.get("object_part", "unknown"),
        "claim_status": state.get("claim_status", "not_enough_information"),
        "claim_status_justification": state.get("claim_status_justification", ""),
        "supporting_image_ids": state.get("supporting_image_ids", ["none"]),
        "valid_image": state.get("valid_image", False),
        "severity": state.get("severity", "unknown"),
    }
    validated = validate_row(row, claim_object=state.get("claim_object", ""))
    writer.writerow(validated)


def close_writer(writer: csv.DictWriter, fh) -> None:  # noqa: ANN001
    fh.close()

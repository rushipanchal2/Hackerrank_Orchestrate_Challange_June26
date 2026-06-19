"""LangGraph state schema for one claim evaluation."""

import operator
from typing import Annotated, Any, TypedDict


class ClaimState(TypedDict):
    # ── inputs (set once from CSV row, never mutated) ──────────────────────
    user_id: str
    image_paths: list[str]          # CSV path strings split on ";"
    image_paths_raw: str            # original semicolon-joined string (for output echo)
    user_claim: str                 # raw multi-turn chat transcript
    claim_object: str               # "car" | "laptop" | "package"

    # ── reference data (populated by load_context) ─────────────────────────
    user_history: dict[str, Any]    # matching row from user_history.csv; {} if not found
    applicable_requirements: list[dict[str, str]]   # filtered evidence_requirements rows
    encoded_images: list[dict]      # [{image_id, base64_str, path, exists}]

    # ── intermediate: claim extraction ─────────────────────────────────────
    normalized_claim: str           # English, concise
    claimed_parts: list[str]        # e.g. ["front_bumper", "headlight"]

    # ── intermediate: image analysis ───────────────────────────────────────
    image_analyses: list[dict]      # [{image_id, quality_flags, content_summary,
                                    #   matches_claim_object, matches_claimed_part,
                                    #   issue_visible, non_original_image,
                                    #   possible_manipulation, injection_text_present}]
    injection_detected: bool        # any injection instruction found

    # ── outputs (set by synthesize_decision or make_fallback_decision) ─────
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: list[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: list[str]
    valid_image: bool
    severity: str

    # ── error tracking: append-only across nodes ───────────────────────────
    errors: Annotated[list[str], operator.add]


def default_state(
    user_id: str,
    image_paths_raw: str,
    user_claim: str,
    claim_object: str,
) -> ClaimState:
    """Build the initial state dict from a CSV row."""
    return ClaimState(
        user_id=user_id,
        image_paths=[p.strip() for p in image_paths_raw.split(";") if p.strip()],
        image_paths_raw=image_paths_raw,
        user_claim=user_claim,
        claim_object=claim_object,
        user_history={},
        applicable_requirements=[],
        encoded_images=[],
        normalized_claim="",
        claimed_parts=[],
        image_analyses=[],
        injection_detected=False,
        evidence_standard_met=False,
        evidence_standard_met_reason="",
        risk_flags=[],
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification="",
        supporting_image_ids=["none"],
        valid_image=False,
        severity="unknown",
        errors=[],
    )

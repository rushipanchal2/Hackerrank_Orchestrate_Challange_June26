"""LangGraph node functions for the claim evaluation pipeline."""

import json
import re

from langgraph.types import RunnableConfig

from code.graph.state import ClaimState
from code.prompts.analyze_images import ANALYZE_IMAGES_SYSTEM
from code.prompts.extract_claim import EXTRACT_CLAIM_SYSTEM
from code.prompts.synthesize_decision import SYNTHESIZE_DECISION_SYSTEM
from code.utils import llm
from code.utils.image_loader import encode_images
from code.utils.schema import PARTS_LIST_BY_OBJECT


# ── JSON extraction (3-strategy, robust for all free models) ─────────────────

def _repair_json(text: str) -> str:
    """Fix common model JSON mistakes: unquoted string values after string keys."""
    # Fix: "key": unquoted value, → "key": "unquoted value",
    # Matches string keys whose value starts without a quote, bool, number, [ or {
    text = re.sub(
        r'("(?:[^"\\]|\\.)*"\s*:\s*)([^",\[\]{}0-9tfn\s\-][^,\n\]}\'"]*)',
        lambda m: m.group(1) + '"' + m.group(2).strip().rstrip(',').strip() + '"',
        text,
    )
    return text


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response using 4 fallback strategies.

    Strategy 1: strip markdown fences, parse directly.
    Strategy 2: repair common model mistakes (unquoted strings), then parse.
    Strategy 3: find the first {...} block via regex (handles prose prefix/suffix).
    Strategy 4: find the last {...} block (some models repeat the schema first).
    """
    text = text.strip()

    # Strategy 1 — strip fences and try direct parse
    clean = re.sub(r"^```(?:json)?\s*", "", text)
    clean = re.sub(r"\s*```\s*$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — repair unquoted string values, then parse
    try:
        return json.loads(_repair_json(clean))
    except json.JSONDecodeError:
        pass

    # Strategy 3 — extract first balanced {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        block = brace_match.group()
        for candidate in (block, _repair_json(block)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 4 — find last {...} block (model repeated schema then answered)
    all_blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for block in reversed(all_blocks):
        for candidate in (block, _repair_json(block)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON found in model response. Response was:\n{text[:300]}")


# ── LLM call wrappers ─────────────────────────────────────────────────────────

def _text_call(system: str, user: str, max_tokens: int = 1024) -> str:
    return llm.text_call(system, user, max_tokens)


def _vision_call(system: str, images: list, text: str,
                 max_tokens: int = 2048) -> str:
    return llm.vision_call(system, images, text, max_tokens)


# ── Node 1: load_context ──────────────────────────────────────────────────────

def load_context(state: ClaimState, config: RunnableConfig) -> dict:
    """Fetch user history, filter requirements, encode images."""
    cfg = config.get("configurable", {})
    user_history_dict: dict = cfg.get("user_history_dict", {})
    requirements_list: list = cfg.get("requirements_list", [])
    images_base_dir: str = cfg.get("images_base_dir", "dataset")

    user_history = user_history_dict.get(state["user_id"], {})

    applicable_requirements = [
        r for r in requirements_list
        if r.get("claim_object") in (state["claim_object"], "all")
    ]

    encoded = encode_images(state["image_paths"], images_base_dir)

    errors: list[str] = []
    for img in encoded:
        if not img["exists"]:
            errors.append(f"Missing image: {img['path']}")

    return {
        "user_history": user_history,
        "applicable_requirements": applicable_requirements,
        "encoded_images": encoded,
        "errors": errors,
    }


# ── Node 2: extract_claim ─────────────────────────────────────────────────────

def extract_claim(state: ClaimState) -> dict:
    """Normalise multilingual claim conversation → English + claimed parts list."""
    allowed_parts = PARTS_LIST_BY_OBJECT.get(state["claim_object"], "unknown")
    user_content = (
        f"Claim object: {state['claim_object']}\n"
        f"Allowed parts: {allowed_parts}\n\n"
        f"Conversation transcript:\n{state['user_claim']}"
    )

    try:
        raw = _text_call(EXTRACT_CLAIM_SYSTEM, user_content, max_tokens=256)
        result = _parse_json(raw)
        return {
            "normalized_claim": result.get("normalized_claim", "")[:300],
            "claimed_parts": result.get("claimed_parts", ["unknown"]),
        }
    except Exception as exc:
        return {
            "normalized_claim": state["user_claim"][:200],
            "claimed_parts": ["unknown"],
            "errors": [f"extract_claim failed: {exc}"],
        }


# ── Node 3: analyze_images ────────────────────────────────────────────────────

def analyze_images(state: ClaimState) -> dict:
    """Per-image visual analysis + injection detection via vision model."""
    usable = [img for img in state["encoded_images"] if img["exists"]]

    if not usable:
        return {
            "image_analyses": [],
            "injection_detected": False,
            "errors": ["No usable images to analyze."],
        }

    analysis_text = (
        f"Claim object: {state['claim_object']}\n"
        f"Normalized claim: {state['normalized_claim']}\n"
        f"Claimed parts: {', '.join(state['claimed_parts'])}\n\n"
        f"Conversation transcript (check for injection only):\n"
        f"{state['user_claim'][:800]}\n\n"
        "Analyse each image and return JSON as specified."
    )

    try:
        raw = _vision_call(ANALYZE_IMAGES_SYSTEM, usable, analysis_text,
                           max_tokens=768)
        result = _parse_json(raw)
        return {
            "image_analyses": result.get("image_analyses", []),
            "injection_detected": bool(result.get("injection_detected", False)),
        }
    except Exception as exc:
        return {
            "image_analyses": [],
            "injection_detected": False,
            "errors": [f"analyze_images failed: {exc}"],
        }


# ── Conditional edge helper ───────────────────────────────────────────────────

def route_on_images(state: ClaimState) -> str:
    if any(img.get("exists") for img in state["encoded_images"]):
        return "synthesize_decision"
    return "make_fallback_decision"


# ── Node 4: make_fallback_decision ────────────────────────────────────────────

def make_fallback_decision(state: ClaimState) -> dict:
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "No usable images were submitted or could be loaded.",
        "risk_flags": ["damage_not_visible"],
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "No images could be loaded for review.",
        "supporting_image_ids": ["none"],
        "valid_image": False,
        "severity": "unknown",
    }


# ── Node 5: synthesize_decision ───────────────────────────────────────────────

def synthesize_decision(state: ClaimState) -> dict:
    """Merge image analysis + user history + requirements → final verdict."""
    reqs_text = "\n".join(
        f"- [{r['requirement_id']}] {r['applies_to']}: {r['minimum_image_evidence']}"
        for r in state["applicable_requirements"]
    )

    history = state["user_history"]
    history_text = (
        f"past_claim_count={history.get('past_claim_count', 0)}, "
        f"accepted={history.get('accept_claim', 0)}, "
        f"rejected={history.get('rejected_claim', 0)}, "
        f"last_90_days={history.get('last_90_days_claim_count', 0)}, "
        f"flags={history.get('history_flags', 'none')}, "
        f"summary={history.get('history_summary', 'No history.')}"
    ) if history else "No user history found (new user)."

    analyses_text = json.dumps(state["image_analyses"], indent=2)

    user_content = (
        f"CLAIM OBJECT: {state['claim_object']}\n"
        f"NORMALIZED CLAIM: {state['normalized_claim']}\n"
        f"CLAIMED PARTS: {', '.join(state['claimed_parts'])}\n"
        f"INJECTION DETECTED: {state['injection_detected']}\n\n"
        f"IMAGE ANALYSES:\n{analyses_text}\n\n"
        f"APPLICABLE EVIDENCE REQUIREMENTS:\n{reqs_text or '(none found)'}\n\n"
        f"USER HISTORY: {history_text}\n\n"
        "Produce the final claim decision as JSON."
    )

    try:
        raw = _text_call(SYNTHESIZE_DECISION_SYSTEM, user_content, max_tokens=512)
        result = _parse_json(raw)
        return {
            "evidence_standard_met": bool(result.get("evidence_standard_met", False)),
            "evidence_standard_met_reason": result.get("evidence_standard_met_reason", ""),
            "risk_flags": result.get("risk_flags", ["none"]),
            "issue_type": result.get("issue_type", "unknown"),
            "object_part": result.get("object_part", "unknown"),
            "claim_status": result.get("claim_status", "not_enough_information"),
            "claim_status_justification": result.get("claim_status_justification", ""),
            "supporting_image_ids": result.get("supporting_image_ids", ["none"]),
            "valid_image": bool(result.get("valid_image", False)),
            "severity": result.get("severity", "unknown"),
        }
    except Exception as exc:
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Decision synthesis failed.",
            "risk_flags": ["manual_review_required"],
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "Automated review failed; manual review required.",
            "supporting_image_ids": ["none"],
            "valid_image": False,
            "severity": "unknown",
            "errors": [f"synthesize_decision failed: {exc}"],
        }


# ── Node 6: format_output ─────────────────────────────────────────────────────

def format_output(state: ClaimState) -> dict:
    """Validate all output fields against allowed-value lists. No LLM call."""
    from code.utils.schema import validate_row

    row = {
        "claim_status": state.get("claim_status", "not_enough_information"),
        "issue_type": state.get("issue_type", "unknown"),
        "object_part": state.get("object_part", "unknown"),
        "severity": state.get("severity", "unknown"),
        "risk_flags": state.get("risk_flags", ["none"]),
        "supporting_image_ids": state.get("supporting_image_ids", ["none"]),
        "evidence_standard_met": state.get("evidence_standard_met", False),
        "valid_image": state.get("valid_image", False),
    }
    validated = validate_row(row, claim_object=state.get("claim_object", ""))

    return {
        "claim_status": validated["claim_status"],
        "issue_type": validated["issue_type"],
        "object_part": validated["object_part"],
        "severity": validated["severity"],
        "risk_flags": validated["risk_flags"].split(";"),
        "supporting_image_ids": validated["supporting_image_ids"].split(";"),
        "evidence_standard_met": validated["evidence_standard_met"] == "true",
        "valid_image": validated["valid_image"] == "true",
    }

"""Allowed output values and row validation."""

REQUIRED_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

PARTS_BY_OBJECT = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

PARTS_LIST_BY_OBJECT = {
    "car": "front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown",
    "laptop": "screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown",
    "package": "box, package_corner, package_side, seal, label, contents, item, unknown",
}


def validate_row(row: dict, claim_object: str = "") -> dict:
    """Validate and normalise all output fields. Returns a new dict with safe values."""
    result = dict(row)

    if result.get("claim_status") not in ALLOWED_CLAIM_STATUS:
        result["claim_status"] = "not_enough_information"

    if result.get("issue_type") not in ALLOWED_ISSUE_TYPE:
        result["issue_type"] = "unknown"

    if result.get("severity") not in ALLOWED_SEVERITY:
        result["severity"] = "unknown"

    allowed_parts = PARTS_BY_OBJECT.get(claim_object, set())
    if result.get("object_part") not in allowed_parts:
        result["object_part"] = "unknown"

    # Normalise risk_flags → validated semicolon-separated string
    raw_flags = result.get("risk_flags", "none")
    if isinstance(raw_flags, list):
        flags = raw_flags
    else:
        flags = [f.strip() for f in str(raw_flags).split(";") if f.strip()]
    valid_flags = [f for f in flags if f in ALLOWED_RISK_FLAGS]
    result["risk_flags"] = ";".join(valid_flags) if valid_flags else "none"

    # Normalise supporting_image_ids → semicolon string
    img_ids = result.get("supporting_image_ids", "none")
    if isinstance(img_ids, list):
        result["supporting_image_ids"] = ";".join(img_ids) if img_ids else "none"

    # Normalise booleans
    for field in ("evidence_standard_met", "valid_image"):
        val = result.get(field, False)
        if isinstance(val, bool):
            result[field] = "true" if val else "false"
        else:
            result[field] = "true" if str(val).lower() == "true" else "false"

    return result

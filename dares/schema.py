from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Model output does not contain a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ValueError(f"Model output contains invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def validate_decision(value: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("target", "objective", "anchor", "decisions") if key not in value]
    if missing:
        raise ValueError(f"Decider output is missing: {', '.join(missing)}")
    anchor = value["anchor"]
    if not isinstance(anchor, dict) or "pattern" not in anchor or not isinstance(anchor.get("entities"), list):
        raise ValueError("Decider anchor must contain pattern and entities")
    if not isinstance(value["decisions"], list):
        raise ValueError("Decider decisions must be a list")
    for decision in value["decisions"]:
        if not isinstance(decision, dict) or not all(key in decision for key in ("type", "requirement", "entities")):
            raise ValueError("Each decision must contain type, requirement, and entities")
    return value


def validate_selection(value: dict[str, Any], candidate_count: int) -> list[str]:
    ids = value.get("selected_block_ids")
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("Selector output must contain a selected_block_ids string list")
    valid = {f"B{index}" for index in range(candidate_count)}
    result = []
    for block_id in ids:
        if block_id not in valid:
            raise ValueError(f"Unknown candidate identifier: {block_id}")
        if block_id not in result:
            result.append(block_id)
    return result

import json
from pathlib import Path

import pytest

from dares.io import candidate_text, normalize_task
from dares.prompts import load_template, render_decider, render_selector
from dares.schema import extract_json_object, validate_decision, validate_selection


def task():
    return normalize_task({"task_id": "1", "project": "repo", "path": "a.py",
                           "input": "def f():\n", "All_contexts": [{"path": "b.py", "code": "x = 1"}]}, 0)


def test_legacy_input_and_prompt_rendering():
    value = task()
    assert value["repository"] == "repo"
    assert candidate_text(value["candidate_blocks"][0]).startswith("# b.py")
    decider = render_decider(load_template(None, "decider.txt"), value)
    selector = render_selector(load_template(None, "selector.txt"), value, "{}", ["x = 1"])
    assert "{REPOSITORY_NAME}" not in decider
    assert '<candidate_block id="B0">' in selector


def test_schemas():
    decision = {"target": "f", "objective": "o", "anchor": {"pattern": "p", "entities": []},
                "decisions": [{"type": "t", "requirement": "r", "entities": []}]}
    assert validate_decision(extract_json_object(json.dumps(decision))) == decision
    assert validate_selection({"selected_block_ids": ["B1", "B0", "B1"]}, 2) == ["B1", "B0"]
    with pytest.raises(ValueError):
        validate_selection({"selected_block_ids": ["B2"]}, 2)

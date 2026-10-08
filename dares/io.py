from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _first(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def normalize_task(row: dict[str, Any], index: int) -> dict[str, Any]:
    task_id = str(_first(row, ("id", "task_id", "namespace"), index))
    repository = str(_first(row, ("repository", "project", "repo"), ""))
    current_file = str(_first(row, ("current_file", "path", "file_path"), ""))
    cursor_prefix = str(_first(row, ("cursor_prefix", "input", "prefix"), ""))
    candidates = _first(row, ("candidate_blocks", "All_contexts", "all_contexts"), [])
    if not isinstance(candidates, list):
        raise ValueError(f"Task {task_id}: candidate pool must be a list")
    if not repository or not cursor_prefix:
        raise ValueError(f"Task {task_id}: repository and cursor prefix are required")
    return {"id": task_id, "repository": repository, "current_file": current_file,
            "cursor_prefix": cursor_prefix, "candidate_blocks": candidates, "source": row}


def candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if not isinstance(candidate, dict):
        return str(candidate)
    code = _first(candidate, ("code", "content", "text", "snippet"), "")
    path = _first(candidate, ("path", "file_path", "file", "filename"), "")
    name = _first(candidate, ("name", "qualified_name", "symbol"), "")
    header = " :: ".join(str(value) for value in (path, name) if value)
    return f"# {header}\n{code}" if header else str(code)

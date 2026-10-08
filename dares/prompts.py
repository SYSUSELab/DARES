from __future__ import annotations

from pathlib import Path
import sysconfig
from typing import Any


def _prompt_dir() -> Path:
    repository_prompts = Path(__file__).resolve().parents[1] / "prompts"
    if repository_prompts.is_dir():
        return repository_prompts
    return Path(sysconfig.get_path("data")) / "share" / "dares" / "prompts"


def load_template(path: str | Path | None, default_name: str) -> str:
    prompt_path = Path(path) if path else _prompt_dir() / default_name
    return prompt_path.read_text(encoding="utf-8")


def render_decider(template: str, task: dict[str, Any]) -> str:
    return (template.replace("{REPOSITORY_NAME}", task["repository"])
            .replace("{CURRENT_FILE_PATH}", task["current_file"])
            .replace("{CODE_BEFORE_COMPLETION_POSITION}", task["cursor_prefix"]))


def render_selector(template: str, task: dict[str, Any], stage1_output: str,
                    candidate_texts: list[str]) -> str:
    blocks = "\n".join(
        f'<candidate_block id="B{index}">\n{text}\n</candidate_block>'
        for index, text in enumerate(candidate_texts)
    )
    return (template.replace("{REPOSITORY_NAME}", task["repository"])
            .replace("{CURRENT_FILE_PATH}", task["current_file"])
            .replace("{CODE_BEFORE_COMPLETION_POSITION}", task["cursor_prefix"])
            .replace("{STAGE_1_OUTPUT}", stage1_output)
            .replace("{CANDIDATE_BLOCKS}", blocks))

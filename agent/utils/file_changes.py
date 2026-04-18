"""Shared helper for applying code/test file changes from LLM output."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ApplyChangesError(Exception):
    """Raised when a change cannot be applied (bad edit, missing file, etc.)."""


def apply_changes(
    repo_path: Path, changes: list[dict[str, Any]]
) -> tuple[int, list[str]]:
    """Apply a list of file changes, handling both `create` and `edit` operations.

    Returns (files_written, files_skipped_as_noop).
    Raises ApplyChangesError on any invalid edit (missing file, old_string not found,
    or old_string ambiguous).
    """
    written = 0
    skipped: list[str] = []
    for change in changes:
        path_str = change.get("path", "")
        if not path_str:
            continue
        file_path = repo_path / path_str
        operation = change.get("operation") or (
            "edit" if file_path.exists() and change.get("edits") else "create"
        )

        if operation == "edit":
            if not file_path.exists():
                raise ApplyChangesError(
                    f"Edit operation requested for non-existent file: {path_str}"
                )
            content = file_path.read_text(errors="replace")
            original = content
            for edit in change.get("edits", []):
                old_s = edit.get("old_string", "")
                new_s = edit.get("new_string", "")
                if not old_s:
                    raise ApplyChangesError(
                        f"Empty old_string in edit for {path_str}"
                    )
                occurrences = content.count(old_s)
                if occurrences == 0:
                    raise ApplyChangesError(
                        f"Edit failed for {path_str}: old_string not found. "
                        f"First 200 chars: {old_s[:200]!r}"
                    )
                if occurrences > 1:
                    raise ApplyChangesError(
                        f"Edit failed for {path_str}: old_string matches "
                        f"{occurrences} times; include more surrounding context."
                    )
                content = content.replace(old_s, new_s, 1)
            if content == original:
                skipped.append(path_str)
                continue
            file_path.write_text(content)
            written += 1
        else:  # create
            new_content = change.get("new_content", "")
            if file_path.exists() and file_path.read_text(errors="replace") == new_content:
                skipped.append(path_str)
                continue
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_content)
            written += 1
    return written, skipped

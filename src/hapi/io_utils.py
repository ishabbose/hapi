"""Shared helpers for incremental runs."""

from __future__ import annotations

from pathlib import Path


def should_skip(output: Path, force: bool) -> bool:
    if force:
        return False
    if output.exists():
        print(f"Skipping — already exists: {output}")
        print("Re-run with --force to overwrite.")
        return True
    return False


def local_tag(tag: str) -> str:
    return tag.split("}")[-1]

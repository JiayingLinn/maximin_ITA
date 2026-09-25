#!/usr/bin/env python3
"""Check a source-only release for common identifying or generated material.

This is a conservative static check, not proof that arbitrary text is anonymous.
Findings report locations and categories without echoing potentially secret text.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

PATTERNS = {
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "user or cluster absolute path": re.compile(
        r"/(?:Users|home|gpfs|scratch|mnt|lustre)/[A-Za-z0-9_.-]+"
        r"|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+"
    ),
    "access token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "literal credential": re.compile(
        r"(?i)(?:api_key|access_token|password)\s*[=:]\s*[\"'](?![<${])[A-Za-z0-9_/-]{12,}[\"']"
    ),
}
ALLOWED_SUFFIXES = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".jinja"}
FORBIDDEN_DIRS = {
    ".venv", "__pycache__", ".cache", ".hf_cache", "wandb", "outputs", "results",
    "data", "datasets", "models", "checkpoints", ".codex", ".claude", ".agents",
}


def audit(root: Path) -> list[str]:
    findings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        # Git history is managed separately; it is not part of this source scan.
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            findings.append(f"{relative}: symbolic link")
            continue
        if path.is_dir():
            if path.name in FORBIDDEN_DIRS:
                findings.append(f"{relative}: generated/private directory")
            continue
        if path.name != ".gitignore" and path.suffix not in ALLOWED_SUFFIXES:
            findings.append(f"{relative}: non-source file type")
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            findings.append(f"{relative}: unreadable text or binary file")
            continue
        for line_number, line in enumerate(contents.splitlines(), 1):
            for category, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {category}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    findings = audit(args.root)
    if findings:
        print("Release audit failed:")
        print("\n".join(findings))
        return 1
    print("Release audit passed: source text only; no recognized identifying patterns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

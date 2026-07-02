#!/usr/bin/env python3
"""Audit comments and docstrings against the original main baseline."""

from __future__ import annotations

import argparse
import ast
import io
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path


BASELINE = "492f35d"
DIRECTIVES = ("noqa", "type:", "pyright:", "mypy:", "fmt:")
TEXT_SOURCES = {
    "Dockerfile",
    ".gitignore",
}
TEXT_SUFFIXES = {".sh", ".zsh", ".bash", ".ps1"}


def git(*arguments: str) -> bytes:
    """Return bytes from one Git command."""
    return subprocess.check_output(["git", *arguments])


def tree_files(revision: str) -> list[str]:
    """List files tracked at a revision."""
    if revision == "WORKTREE":
        return git(
            "ls-files", "--cached", "--others", "--exclude-standard"
        ).decode().splitlines()
    return git("ls-tree", "-r", "--name-only", revision).decode().splitlines()


def read_revision(revision: str, path: str) -> bytes | None:
    """Read a tracked file or return None when absent."""
    if revision == "WORKTREE":
        candidate = Path(path)
        return candidate.read_bytes() if candidate.is_file() else None
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def python_comments(data: bytes) -> Counter[str]:
    """Count Python comment tokens."""
    try:
        return Counter(
            token.string
            for token in tokenize.tokenize(io.BytesIO(data).readline)
            if token.type == tokenize.COMMENT
        )
    except (SyntaxError, tokenize.TokenError):
        return Counter()


def text_comments(data: bytes) -> Counter[str]:
    """Count full-line comments in tool and shell sources."""
    return Counter(
        line.strip()
        for line in data.decode(errors="replace").splitlines()
        if line.lstrip().startswith("#")
    )


def is_directive(comment: str) -> bool:
    """Return whether a comment is an allowed tool directive."""
    return (
        comment.startswith("#!")
        or comment == "# syntax=docker/dockerfile:1"
        or any(value in comment for value in DIRECTIVES)
    )


def comment_counter(path: str, data: bytes) -> Counter[str]:
    """Select the comment parser for a source path."""
    return (
        python_comments(data)
        if path.endswith(".py")
        else text_comments(data)
    )


def source_path(path: str) -> bool:
    """Return whether a path is covered by the source-comment policy."""
    candidate = Path(path)
    return (
        candidate.suffix == ".py"
        or candidate.suffix in TEXT_SUFFIXES
        or candidate.name in TEXT_SOURCES
    )


def audit_comments(revision: str) -> list[str]:
    """Return comment-policy violations for a revision."""
    baseline_files = set(tree_files(BASELINE))
    revision_files = set(tree_files(revision))
    violations: list[str] = []
    for path in sorted(baseline_files | revision_files):
        if not source_path(path):
            continue
        baseline_data = read_revision(BASELINE, path)
        revision_data = read_revision(revision, path)
        baseline = (
            comment_counter(path, baseline_data)
            if baseline_data is not None
            else Counter()
        )
        current = (
            comment_counter(path, revision_data)
            if revision_data is not None
            else Counter()
        )
        for comment, count in (baseline - current).items():
            violations.append(
                f"{path}: missing baseline comment {count}x {comment}"
            )
        for comment, count in (current - baseline).items():
            if not is_directive(comment):
                violations.append(
                    f"{path}: new comment {count}x {comment}"
                )
    return violations


def audit_docstrings(revision: str) -> list[str]:
    """Return malformed docstrings in files changed from baseline."""
    comparison = (
        (BASELINE,)
        if revision == "WORKTREE"
        else (f"{BASELINE}..{revision}",)
    )
    changed = git(
        "diff", "--name-only", *comparison, "--", "*.py"
    ).decode().splitlines()
    violations: list[str] = []
    for path in changed:
        data = read_revision(revision, path)
        if data is None:
            continue
        try:
            tree = ast.parse(data.decode())
        except SyntaxError as exc:
            violations.append(f"{path}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring is None:
                continue
            stripped = docstring.strip()
            if (
                "\n" in stripped
                or len(stripped) > 140
                or not stripped.endswith((".", "!", "?"))
                or stripped in {"Args:.", "Returns:.", "Observation:."}
                or ",." in stripped
                or ":." in stripped
            ):
                violations.append(
                    f"{path}:{node.lineno}: malformed docstring for {node.name}"
                )
    return violations


def main() -> int:
    """Run the source-policy audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("revision", nargs="?", default="WORKTREE")
    args = parser.parse_args()
    violations = [
        *audit_comments(args.revision),
        *audit_docstrings(args.revision),
    ]
    if violations:
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
